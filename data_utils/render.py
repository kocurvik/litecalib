"""Render a view out of an equirectangular panorama."""

import math

import torch
import torch.nn.functional as F


from litecalib.anycalib.cameras import CameraFactory

__all__ = ["rotation", "view_rays", "sample_view", "to_uint8"]


def rotation(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """``Rz(roll) @ Rx(pitch) @ Ry(yaw)``, AnyCalib's world-from-camera convention (float64)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = torch.tensor([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]], dtype=torch.float64)
    Rx = torch.tensor([[1, 0, 0], [0, cp, sp], [0, -sp, cp]], dtype=torch.float64)
    Ry = torch.tensor([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]], dtype=torch.float64)
    return Rz @ Rx @ Ry


def view_rays(
    cam_id: str, params: torch.Tensor, h: int, w: int, roll: float, pitch: float, yaw: float,
    offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Longitude and latitude (h, w), float64, of every view pixel's ray."""
    cam = CameraFactory.create_from_id(cam_id)
    bearings, valid = cam.ray_grid(h, w, params.to(torch.float64), offset=offset)
    if valid is not None and not bool(valid.all()):
        raise ValueError(f"{cam_id} {params.tolist()}: some pixels have no ray")
    rays = bearings.reshape(h * w, 3) @ rotation(roll, pitch, yaw)
    lon = torch.atan2(rays[:, 0], rays[:, 2])
    lat = torch.atan2(rays[:, 1], torch.linalg.norm(rays[:, [0, 2]], dim=-1))
    return lon.reshape(h, w), lat.reshape(h, w)


def sample_view(pano: torch.Tensor, lon: torch.Tensor, lat: torch.Tensor, scale: float) -> torch.Tensor:
    """(3, h, w) float view in [0, 1], sampled from a (3, H, W) uint8 panorama."""
    H, W = pano.shape[-2:]
    u = (lon + math.pi) / (2 * math.pi) * W - 0.5
    v = (lat + math.pi / 2) / math.pi * H - 0.5
    margin = 2 if scale >= 1 else math.ceil(2 / scale) + 2

    uc = float(u[u.shape[0] // 2, u.shape[1] // 2])
    u = uc + torch.remainder(u - uc + W / 2, W) - W / 2
    c0 = math.floor(float(u.min())) - margin
    c1 = math.ceil(float(u.max())) + margin + 1
    if c1 - c0 >= W:
        u = torch.remainder(u, W)
        c0, c1 = -margin, W + margin
    r0 = max(math.floor(float(v.min())) - margin, 0)
    r1 = min(math.ceil(float(v.max())) + margin + 1, H)

    cols = torch.arange(c0, c1) % W
    crop = pano[:, r0:r1].index_select(-1, cols).unsqueeze(0).float().div_(255)
    x, y = u - c0, v - r0

    if scale >= 1:
        out = F.grid_sample(crop, _grid(x, y, crop), mode="bicubic", padding_mode="border",
                            align_corners=True)
        return out[0].clamp_(0, 1)
    ch, cw = crop.shape[-2:]
    size = (max(1, round(ch * scale)), max(1, round(cw * scale)))
    small = F.interpolate(crop, size=size, mode="bicubic", align_corners=False, antialias=True)
    del crop
    small = small.clamp_(0, 1)
    x = (x + 0.5) * (size[1] / cw) - 0.5
    y = (y + 0.5) * (size[0] / ch) - 0.5
    out = F.grid_sample(small, _grid(x, y, small), mode="bilinear", padding_mode="border",
                        align_corners=True)
    return out[0]


def to_uint8(img: torch.Tensor) -> "torch.Tensor":
    """(3, h, w) in [0, 1] -> (h, w, 3) uint8, rounded (upstream truncates)."""
    return img.mul(255).round_().clamp_(0, 255).to(torch.uint8).permute(1, 2, 0)


def _grid(x: torch.Tensor, y: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """(1, h, w, 2) normalized grid for `grid_sample(align_corners=True)`."""
    hh, ww = img.shape[-2:]
    gx = 2 * x / max(ww - 1, 1) - 1
    gy = 2 * y / max(hh - 1, 1) - 1
    return torch.stack((gx, gy), dim=-1).unsqueeze(0).to(img.dtype)

"""AnyCalib's dense supervision: ground-truth rays at every input pixel and the `l1-z1` loss."""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litecalib import geometry

from litecalib.anycalib.cameras import CameraFactory
from litecalib.anycalib.manifolds import Unit3

RAD2DEG = 180 / math.pi


def gt_rays(batch: dict, h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground-truth unit rays at the pixel centers of the network input, and their validity."""
    rays, valid = [], []
    for cam_id, params, n in zip(batch["cam_id"], batch["intrinsics"], batch["nparams"].tolist()):
        cam = CameraFactory.create_from_id(cam_id)
        r, v = cam.ray_grid(h, w, params[:n].to(device, torch.float32), 0.5)
        rays.append(r.reshape(h * w, 3))
        valid.append(torch.ones(h * w, dtype=torch.bool, device=device) if v is None
                     else v.reshape(h * w))
    rays, valid = torch.stack(rays), torch.stack(valid)
    return rays, valid & rays.isfinite().all(-1)


def field_loss(pred: dict, rays: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """AnyCalib's `l1-z1` over the valid pixels of the batch, and the per-image angular error."""
    gt_tangent = Unit3.logmap_at_z1(torch.where(valid[..., None], rays, rays.new_tensor((0, 0, 1.0))))
    err = (pred["tangent_coords"] - gt_tangent).abs().sum(-1) * valid
    loss = err.sum() / valid.sum().clamp(min=1)
    with torch.no_grad():
        safe = torch.where(valid[..., None], rays, pred["rays"].detach())
        ang = Unit3.distance(safe, pred["rays"].detach()) * valid
        per_image = RAD2DEG * ang.sum(-1) / valid.sum(-1).clamp(min=1)
    return loss, {"ang_err_deg": per_image}


def dense_loss(tangent: torch.Tensor, batch: dict, h: int, w: int,
               repr: str = "logmap") -> tuple[torch.Tensor, dict]:
    """`field_loss` on a (B, H*W, 2) tangent field (`model.RadialProfileNet(dense_head=True)`), against `batch`'s
    cameras at its (H, W) input: the loss and the batch's mean per-image angular error (deg), as
    `train/train.py` logs it (``dense_deg``).
    """
    tangent = tangent.float()
    rays, valid = gt_rays(batch, h, w, tangent.device)
    if repr == "logmap":
        pred = {"tangent_coords": tangent, "rays": Unit3.expmap_at_z1(tangent)}
        loss, m = field_loss(pred, rays, valid)
        return loss, {"dense_deg": float(m["ang_err_deg"].mean())}
    assert repr == "fov", f"unknown dense representation {repr!r}"
    safe = torch.where(valid[..., None], rays, rays.new_tensor((0, 0, 1.0)))
    err = (tangent - geometry.rays_fov(safe)).abs().sum(-1) * valid
    loss = err.sum() / valid.sum().clamp(min=1)
    with torch.no_grad():
        pred_rays = geometry.fov_rays(tangent.detach())
        ang = Unit3.distance(safe, pred_rays) * valid
        per_image = RAD2DEG * ang.sum(-1) / valid.sum(-1).clamp(min=1)
    return loss, {"dense_deg": float(per_image.mean())}

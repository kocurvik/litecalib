"""Calibration from images: preprocessing, the network, AnyCalib's calibrator on the predicted rays, and the
resulting `Calibration` (K, distortion, undistortion, OpenCV export)."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from litecalib import geometry
from litecalib.anycalib.cameras import CameraFactory
from litecalib.model import RadialProfileNet, build_anycalib

RESOLUTION = 102_400
AR_RANGE = (0.5, 2.0)
FIT_MODELS = ("pinhole", "radial", "kb", "ucm", "division")


def target_size(h: int, w: int, div: int = 14, resolution: int = RESOLUTION) -> tuple[int, int]:
    """The network input size for an ``h x w`` image: ~`resolution` pixels, sides divisible by `div`,
    aspect ratio clamped to `AR_RANGE`."""
    ar = max(AR_RANGE[0], min(h / w, AR_RANGE[1]))
    wt = (resolution / ar) ** 0.5
    ht = ar * wt
    return round(ht / div) * div, round(wt / div) * div


def crop_resize_plan(h: int, w: int, ht: int, wt: int) -> tuple[tuple[int, int, int, int], tuple[float, float]]:
    """Center crop to the target aspect ratio, then resize: ``(top, left, crop_h, crop_w), (sx, sy)``."""
    if w * ht > h * wt:
        crop_h, crop_w = h, round(h * wt / ht)
    else:
        crop_h, crop_w = round(w * ht / wt), w
    crop_h, crop_w = min(crop_h, h), min(crop_w, w)
    top, left = (h - crop_h) // 2, (w - crop_w) // 2
    return (top, left, crop_h, crop_w), (wt / crop_w, ht / crop_h)


def field_rays(values: np.ndarray) -> np.ndarray:
    """(B, 2 K^2) FoV field ``(atan2(X, Z), atan2(Y, Z))`` -> (B, K^2, 3) unit rays."""
    n = values.shape[-1] // 2
    px, py = values[..., :n].astype(np.float64), values[..., n:].astype(np.float64)
    sx, cx, sy, cy = np.sin(px), np.cos(px), np.sin(py), np.cos(py)
    rays = np.stack((sx * cy, cx * sy, cx * cy), axis=-1)
    rays *= np.where(cx + cy < 0, -1.0, 1.0)[..., None]
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def to_tensor(image) -> Tensor:
    """A path, a PIL image (any mode, converted to RGB), an (H, W, 3) uint8 array, a (3, H, W) / (B, 3, H, W)
    tensor in [0, 1], or a list of any of these (all of one size) -> a (B, 3, H, W) float tensor in [0, 1]."""
    if isinstance(image, (list, tuple)):
        images = [to_tensor(im) for im in image]
        sizes = {tuple(im.shape[-2:]) for im in images}
        if len(sizes) > 1:
            raise ValueError(f"the images of one call must have one size, got {sorted(sizes)}")
        return torch.cat(images)
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGB")
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
        return torch.from_numpy(arr.copy()).permute(2, 0, 1).float()[None] / 255.0
    if not isinstance(image, Tensor):
        arr = np.array(image)
        assert arr.ndim == 3 and arr.shape[-1] == 3, f"expected an (H, W, 3) image, got {arr.shape}"
        image = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return image[None] if image.ndim == 3 else image


def fit_rays(cam_id: str, rays: np.ndarray | Tensor, x: np.ndarray, iters: int, cxcy=None,
             simple: bool = False, device: str = "cpu") -> np.ndarray:
    """AnyCalib's calibrator (compiled) on (B, N, 3) rays at pixels ``x`` (N, 2) -> (B, 4 + nk) intrinsics."""
    if device.startswith("cuda"):
        from litecalib import calibrator_cuda

        rays = torch.as_tensor(rays).float().to(device)
        return calibrator_cuda.fit_rays(cam_id, rays, x, iters, cxcy=cxcy, simple=simple).double().cpu().numpy()
    from litecalib import calibrator_cpu

    rays = rays.detach().cpu().numpy() if isinstance(rays, Tensor) else rays
    return calibrator_cpu.fit_rays(cam_id, np.ascontiguousarray(rays, dtype=np.float64), x, iters,
                                   cxcy=cxcy, simple=simple)


class Calibration:
    """The result of a calibration: the fitted camera model and its parameters for B images (B = 1 for a
    joint fit of several images), in pixels of the original images (origin at the top-left corner of the
    top-left pixel, AnyCalib's convention).

    Attributes:
        cam_id: the camera model, e.g. ``"radial:1"``.
        intrinsics: (B, D) float64 tensor, ``fx, fy, cx, cy`` followed by `distortion`.
        image_size: ``(h, w)`` of the calibrated images.
        num_images: how many images the parameters were fitted to (B, or the joint fit's image count).
        shared: whether one camera was fitted jointly to all images.
    """

    def __init__(self, cam_id: str, intrinsics: Tensor, image_size: tuple[int, int], num_images: int,
                 shared: bool = False):
        self.cam_id = cam_id
        self.intrinsics = intrinsics
        self.image_size = tuple(image_size)
        self.num_images = num_images
        self.shared = shared
        self.camera = CameraFactory.create_from_id(cam_id)

    def __len__(self) -> int:
        return len(self.intrinsics)

    def __getitem__(self, i: int) -> "Calibration":
        """The calibration of image `i` (a batch of one)."""
        return Calibration(self.cam_id, self.intrinsics[i : i + 1], self.image_size, 1, self.shared)

    def __repr__(self) -> str:
        return (f"Calibration(cam_id={self.cam_id!r}, image_size={self.image_size}, cameras={len(self)}, "
                f"shared={self.shared}, intrinsics={self.intrinsics.numpy().round(4).tolist()})")

    @property
    def K(self) -> Tensor:
        """(B, 3, 3) calibration matrices."""
        p = self.intrinsics
        K = torch.zeros(len(p), 3, 3, dtype=p.dtype)
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = p[:, 0], p[:, 1], p[:, 2], p[:, 3], 1.0
        return K

    @property
    def distortion(self) -> Tensor:
        """(B, nk) distortion coefficients, named in `distortion_names` (empty for a pinhole)."""
        return self.intrinsics[:, 4:]

    @property
    def distortion_names(self) -> list[str]:
        """The names of the `distortion` coefficients, e.g. ``["k1", "k2"]`` or ``["xi"]``."""
        if self.camera.NAME == "ucm":
            return ["xi"]
        names = [k for k, i in sorted(self.camera.PARAMS_IDX.items(), key=lambda kv: kv[1]) if i >= 4]
        return names[: self.distortion.shape[1]]

    def fov(self) -> dict[str, Tensor]:
        """Vertical and horizontal fields of view in degrees, (B,) each."""
        h, w = self.image_size
        vfov, _ = self.camera.get_vfov(self.intrinsics, h)
        hfov, _ = self.camera.get_hfov(self.intrinsics, w)
        return {"vfov": torch.rad2deg(vfov), "hfov": torch.rad2deg(hfov)}

    def to_opencv(self) -> tuple[np.ndarray, np.ndarray]:
        """``(K, dist)`` for OpenCV, whose pixel origin is the center of the top-left pixel (``c - 0.5``).

        ``pinhole`` and ``radial:K`` give ``dist = (k1, k2, p1 = 0, p2 = 0, k3)`` for `cv2.undistort` etc.;
        ``kb:4`` gives the four coefficients of `cv2.fisheye`. Other models have no OpenCV equivalent."""
        name, nk = geometry.parse_cam_id(self.cam_id)
        K = self.K.numpy().copy()
        K[:, 0, 2] -= 0.5
        K[:, 1, 2] -= 0.5
        k = self.distortion.numpy()
        if name in ("pinhole", "radial"):
            dist = np.zeros((len(k), 5))
            for j, col in enumerate((0, 1, 4)[:nk]):
                dist[:, col] = k[:, j]
        elif name == "kb" and nk == 4:
            dist = k.copy()
        else:
            raise ValueError(f"{self.cam_id!r} has no OpenCV equivalent (use pinhole, radial:1-3 or kb:4)")
        return K, dist

    def undistorted_K(self, scale: float = 1.0, size: tuple[int, int] | None = None) -> Tensor:
        """(B, 3, 3) calibration matrices of the pinhole images `undistort` makes with the same `scale` and `size`:
        the focal lengths times `scale`; the principal point kept, or at the center of an output of `size` (h, w)."""
        K = self.K.clone()
        K[:, 0, 0] *= scale
        K[:, 1, 1] *= scale
        if size is not None:
            K[:, 0, 2], K[:, 1, 2] = 0.5 * size[1], 0.5 * size[0]
        return K

    def undistort(self, image, scale: float = 1.0, size: tuple[int, int] | None = None) -> Tensor:
        """Resample images of the calibrated size to the pinhole camera `undistorted_K(scale, size)`.

        `scale` < 1 zooms out (keeps more of a wide field of view), > 1 zooms in; `size` (h, w) is the output
        size (default: the input's). Pixels the camera does not see are black. One camera (a joint fit, or a
        single image) is applied to every image given; B cameras need B images, one each. Returns
        (B, 3, h, w) in [0, 1]."""
        image = to_tensor(image)
        if tuple(image.shape[-2:]) != self.image_size:
            raise ValueError(f"the calibration is of {self.image_size} images, got {tuple(image.shape[-2:])}")
        if len(self) not in (1, len(image)):
            raise ValueError(f"{len(self)} cameras for {len(image)} images: pass one image per camera, "
                             "or index the calibration (calib[i].undistort(image))")
        h_in, w_in = self.image_size
        h, w = self.image_size if size is None else size
        K_out = self.undistorted_K(scale, size)
        ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64) + 0.5,
                                torch.arange(w, dtype=torch.float64) + 0.5, indexing="ij")
        out = []
        for i, im in enumerate(image):
            j = min(i, len(self) - 1)
            fx, fy, cx, cy = K_out[j, 0, 0], K_out[j, 1, 1], K_out[j, 0, 2], K_out[j, 1, 2]
            rays = torch.stack(((xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)), dim=-1).reshape(1, -1, 3)
            rays = rays / rays.norm(dim=-1, keepdim=True)
            uv, valid = self.camera.project(self.intrinsics[j : j + 1], rays)
            grid = 2 * uv / uv.new_tensor((w_in, h_in)) - 1
            if valid is not None:
                grid = torch.where(valid[..., None], grid, torch.full_like(grid, -2.0))
            grid = grid.reshape(1, h, w, 2).to(im)
            out.append(F.grid_sample(im[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False))
        return torch.cat(out)

    def to_dict(self) -> dict:
        """Plain Python values, e.g. for JSON."""
        return {"cam_id": self.cam_id, "image_size": list(self.image_size), "num_images": self.num_images,
                "shared": self.shared, "intrinsics": self.intrinsics.tolist(), "K": self.K.tolist(),
                "distortion": self.distortion.tolist(), "distortion_names": self.distortion_names}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        """The inverse of `to_dict`."""
        return cls(d["cam_id"], torch.tensor(d["intrinsics"], dtype=torch.float64), tuple(d["image_size"]),
                   d.get("num_images", len(d["intrinsics"])), d.get("shared", False))


def load_checkpoint(path: str | Path) -> tuple[dict, dict]:
    """A released model (its directory, or its ``.safetensors`` file with ``config.json`` beside it or the
    config in its metadata) or a training checkpoint -> (config, state dict)."""
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        config_file = path.with_name("config.json")
        if config_file.is_file():
            config = json.loads(config_file.read_text())
        else:
            with safe_open(str(path), framework="pt") as f:
                metadata = f.metadata() or {}
            if "config" not in metadata:
                raise ValueError(f"{path}: no config.json beside it and no config in its metadata")
            config = json.loads(metadata["config"])
        return config, load_file(str(path))
    state = torch.load(path, map_location="cpu", weights_only=False)
    args = state.get("args", {})
    if state.get("arch") == "anycalib":
        config = {"arch": "anycalib", "backbone": args.get("backbone", "vitl14"),
                  "dense_dim": args.get("dense_dim") or 256}
    else:
        config = {
            "arch": "litecalib",
            "backbone": args.get("backbone", "vitl14"),
            "K": args.get("K", geometry.DEFAULT_K),
            "placement": args.get("placement", geometry.DEFAULT_PLACEMENT),
            "decoder_layers": args.get("decoder_layers", 2),
            "q_dim": args.get("q_dim"),
            "attn_heads": args.get("attn_heads", 8),
            "profile": args.get("profile", "field"),
            "field_repr": args.get("field_repr") or "fov",
        }
    return config, state["model"]


def build_network(config: dict, state: dict | None = None):
    """The network a config describes, with `state` loaded (the dense branch's weights are dropped)."""
    if config["arch"] == "anycalib":
        net = build_anycalib(config["backbone"], config.get("dense_dim", 256))
    else:
        net = RadialProfileNet(
            backbone=config["backbone"], k=config["K"], placement=config.get("placement", "u2"),
            layers=config.get("decoder_layers", 2), q_dim=config.get("q_dim"),
            num_heads=config.get("attn_heads", 8), profile=config.get("profile", "field"),
            field_repr=config.get("field_repr", "fov"),
        )
    if state is not None:
        state = {k: v for k, v in state.items() if not k.startswith("dense.")}
        net.load_state_dict(state, strict=True)
    return net


class Calibrator:
    """A calibration network with its fit: ``calibrate(images, cam_id)`` -> a `Calibration`."""

    def __init__(self, net, config: dict, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.eval().to(self.device)
        self.config = config
        self.arch = config["arch"]
        if self.arch == "litecalib":
            assert net.profile == "field", "only field checkpoints calibrate through this class"
        self._x_cache: dict = {}

    @classmethod
    def from_pretrained(cls, name: str = "litecalib_vitl", device: str | None = None) -> "Calibrator":
        """A released model from the Hugging Face hub, by name (`litecalib.hub.MODELS`) or repository id."""
        from litecalib.hub import download

        return cls.from_checkpoint(download(name), device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> "Calibrator":
        """A released model (directory or ``.safetensors``) or a training checkpoint."""
        config, state = load_checkpoint(path)
        return cls(build_network(config, state), config, device)

    def _grid_pixels(self, ht: int, wt: int) -> np.ndarray:
        key = (ht, wt)
        if key not in self._x_cache:
            if self.arch == "litecalib":
                x = geometry.grid_knot_pixels(ht, wt, self.net.k, dtype=torch.float64).numpy()
            else:
                ys, xs = np.mgrid[0:ht, 0:wt]
                x = np.stack((xs.ravel() + 0.5, ys.ravel() + 0.5), axis=-1).astype(np.float64)
            self._x_cache[key] = np.ascontiguousarray(x)
        return self._x_cache[key]

    @torch.no_grad()
    def _forward(self, image: Tensor) -> tuple[np.ndarray, int, int, tuple, tuple]:
        """One batch of images of one size -> (B, N, 3) rays, the input size, scale_xy and shift_xy."""
        h, w = image.shape[-2:]
        if self.arch == "anycalib":
            m = self.net
            ar = max(m.AR_RANGE[0], min(h / w, m.AR_RANGE[1]))
            im, scale_xy, shift_xy = m.set_im_size(image, m.compute_target_size(m.RESOLUTION, ar))
            out = m.head(m.decoder(m.backbone(im)))
            b, _, ht, wt = im.shape
            rays = out["rays"].permute(0, 2, 3, 1).reshape(b, ht * wt, 3).float().cpu().numpy()
            return (rays, ht, wt, tuple(float(s) for s in scale_xy.flatten()[:2]),
                    tuple(float(s) for s in shift_xy.flatten()[:2]))
        ht, wt = target_size(h, w, self.net.backbone.input_divisor)
        (top, left, crop_h, crop_w), (sx, sy) = crop_resize_plan(h, w, ht, wt)
        batch = image[..., top : top + crop_h, left : left + crop_w]
        if (crop_h, crop_w) != (ht, wt):
            batch = F.interpolate(batch, (ht, wt), mode="bicubic", align_corners=False,
                                  antialias=sy < 1).clamp(0, 1)
        theta = self.net(batch)["theta"].double()
        if self.net.field_repr == "logmap":
            theta = geometry.field_from_logmap(theta)
        return field_rays(theta.cpu().numpy()), ht, wt, (sx, sy), (-left * sx, -top * sy)

    def predict_rays(self, images, batch_size: int = 8) -> dict:
        """The network's output, before any camera model is fitted.

        Returns ``rays`` (B, N, 3) unit rays (numpy) at the pixels ``x`` (N, 2) of the resized network input
        of size ``(ht, wt)``; ``(h, w)`` the original size; ``scale_xy`` / ``shift_xy`` mapping original pixels
        to input pixels (``x_in = x * scale + shift``)."""
        image = to_tensor(images)
        h, w = image.shape[-2:]
        chunks = []
        for i in range(0, len(image), batch_size):
            rays, ht, wt, scale_xy, shift_xy = self._forward(image[i : i + batch_size].to(self.device))
            chunks.append(rays)
        return {"rays": np.concatenate(chunks), "x": self._grid_pixels(ht, wt), "ht": ht, "wt": wt,
                "h": h, "w": w, "scale_xy": scale_xy, "shift_xy": shift_xy}

    def fit(self, pred: dict, cam_id: str = "pinhole", shared: bool = False, fixed_center: bool | None = None,
            square_pixels: bool | None = None, iters: int = 2, fit_device: str = "cpu") -> Calibration:
        """Fit `cam_id` to `predict_rays`' output: one camera per image, or with `shared` one camera to the rays
        of all images together (images of one camera)."""
        name, _ = geometry.parse_cam_id(cam_id)
        assert name in FIT_MODELS and not cam_id.startswith("simple_"), (
            f"{cam_id!r}: the calibrator fits {FIT_MODELS} (use square_pixels for one focal length)")
        fixed_center = self.arch == "litecalib" if fixed_center is None else fixed_center
        square_pixels = fixed_center if square_pixels is None else square_pixels
        assert fixed_center or not square_pixels, "square_pixels needs a fixed principal point"
        rays, x = pred["rays"], pred["x"]
        n_images = len(rays)
        if shared:
            rays = rays.reshape(1, -1, 3)
            x = np.ascontiguousarray(np.tile(x, (n_images, 1)))
        sx, sy = pred["scale_xy"]
        tx, ty = pred["shift_xy"]
        cxcy = np.array((0.5 * pred["w"] * sx + tx, 0.5 * pred["h"] * sy + ty)) if fixed_center else None
        p = fit_rays(cam_id, rays, x, iters, cxcy=cxcy, simple=square_pixels, device=fit_device)
        p[:, 0] /= sx
        p[:, 1] /= sy
        p[:, 2] = (p[:, 2] - tx) / sx
        p[:, 3] = (p[:, 3] - ty) / sy
        return Calibration(cam_id, torch.from_numpy(p), (pred["h"], pred["w"]), n_images, shared)

    def calibrate(self, images, cam_id: str = "pinhole", shared: bool = False, fixed_center: bool | None = None,
                  square_pixels: bool | None = None, iters: int = 2, fit_device: str = "cpu",
                  batch_size: int = 8) -> Calibration:
        """`predict_rays` + `fit`: calibrate an image, a batch of images (one camera each), or with `shared` one
        camera for all of them. See the README's API section for the arguments."""
        pred = self.predict_rays(images, batch_size)
        return self.fit(pred, cam_id, shared, fixed_center, square_pixels, iters, fit_device)

"""AnyCalib's calibrator on a few pixels of a dense ray field instead of every pixel (the sparse-field study).

Samplings:

    dense             every pixel, AnyCalib's released (PyTorch) calibrator
    egrid[:N]         N x N whole pixels evenly spaced from border to border (corners included), the same
                      calibrator on their own rays
    acnb_dense:it<K>  every pixel, AnyCalib's calibrator compiled (`litecalib.calibrator_cpu` / `_cuda`),
                      exactly K Gauss-Newton iterations
    acnb_egrid[:N]:it<K>  the same on the N x N grid
"""

import re

import numpy as np
import torch
from torch import Tensor

from litecalib.anycalib.model.anycalib_pretrained import get_cam_list
from litecalib.predictor import fit_rays

DENSE = "dense"
DEFAULT_GRID = 7
_SPEC = re.compile(r"^(acnb_)?(egrid|dense)(?::(\d+))?(?::it(\d+))?$")


def parse_sampling(spec: str) -> tuple[str, int | None, int | None]:
    """``egrid:5`` -> ('egrid', 5, None); ``acnb_dense:it2`` -> ('acnb_dense', None, 2)."""
    m = _SPEC.match(spec)
    if m is None:
        raise ValueError(f"unknown sampling {spec!r}: dense, egrid[:N], acnb_dense:it<K>, acnb_egrid[:N]:it<K>")
    acnb, layout, n, iters = m.groups()
    if layout == DENSE and n is not None:
        raise ValueError(f"{spec!r}: dense takes no size")
    if bool(acnb) != (iters is not None):
        raise ValueError(f"{spec!r}: an iteration count goes with (and only with) the compiled acnb_ calibrator")
    n = None if layout == DENSE else int(n or DEFAULT_GRID)
    if n is not None and n < 2:
        raise ValueError(f"{spec!r}: a grid needs at least 2 x 2 pixels")
    return (f"acnb_{layout}" if acnb else layout), n, None if iters is None else int(iters)


def sampling_name(spec: str) -> str:
    """Canonical label for a predictor name: ``egrid:5`` -> 'egrid5x5', ``acnb_dense:it2`` -> 'acnb_dense_it2'."""
    kind, n, iters = parse_sampling(spec)
    return kind + ("" if n is None else f"{n}x{n}") + ("" if iters is None else f"_it{iters}")


def pixel_indices(n: int | None, h: int, w: int) -> Tensor:
    """Row-major indices into an ``h x w`` field: every pixel, or an N x N grid from border to border."""
    if n is None:
        return torch.arange(h * w)
    if n > min(h, w):
        raise ValueError(f"a {n} x {n} pixel grid does not fit a {h} x {w} field")
    i = torch.arange(n, dtype=torch.float64)
    rows, cols = (i * (h - 1) / (n - 1)).round().long(), (i * (w - 1) / (n - 1)).round().long()
    return (rows[:, None] * w + cols[None, :]).reshape(-1)


def pixel_centers(idx: Tensor, w: int, **kw) -> Tensor:
    """(P, 2) centers ``(col + 0.5, row + 0.5)`` of row-major pixel indices, AnyCalib's convention."""
    return torch.stack(((idx % w) + 0.5, (idx // w) + 0.5), dim=-1).to(**kw)


class PixelCalibrator:
    """Drop-in for AnyCalib's `Calibrator` that fits to the rays of an `egrid` sampling only, with the wrapped
    calibrator's linear fit, RANSAC fallback and optimizer."""

    def __init__(self, base, n: int):
        self.base, self.n = base, n
        assert base.rm_borders == 0 and base.sample_size <= 0, "sampled fits replace these"
        self.optimizer = base.optimizer

    def __call__(self, pred: dict, data: dict) -> dict:
        _, _, h, w = data["image"].shape
        rays_all = pred["rays"]
        idx = pixel_indices(self.n, h, w).to(rays_all.device)
        im_coords = pixel_centers(idx, w, device=rays_all.device, dtype=rays_all.dtype)
        rays = rays_all[:, idx]
        obs = pred["tangent_coords"][:, idx] if self.optimizer.res_tangent == "z1" else rays
        cams = get_cam_list(data)
        cxcy = data.get("cxcy", None)
        fix_cxcy = cxcy is not None
        cxcy = [None] * len(cams) if cxcy is None else cxcy
        intrinsics, success = [], []
        for rays_, cam_, cxcy_, obs_ in zip(rays, cams, cxcy, obs):
            if self.base.init_with_sac:
                intrinsics_, _ = self.base.ransac(cam_, im_coords, rays_)
                ok = rays_.new_ones((), dtype=torch.bool)
            else:
                intrinsics_, info = cam_.fit(im_coords, rays_, cxcy_)
                ok = (info == 0) and intrinsics_.isfinite().all()
                if not ok:
                    if not self.base.fallback_to_sac:
                        intrinsics.append(torch.ones_like(intrinsics_))
                        success.append(ok)
                        continue
                    intrinsics_, _ = self.base.ransac(cam_, im_coords, rays_)
            intrinsics_opt, cost0, cost, _ = self.optimizer(cam_, intrinsics_, im_coords, obs_, None, fix_cxcy)
            if cost <= cost0:
                intrinsics_ = intrinsics_opt
            intrinsics.append(intrinsics_)
            success.append(ok and cost < cost0)
        return {"intrinsics": intrinsics, "success": torch.stack([torch.as_tensor(s) for s in success])}


class CompiledCalibrator:
    """AnyCalib's calibrator compiled (CPU Numba or CUDA), on every pixel or an `egrid` sampling, exactly
    `iters` Gauss-Newton iterations; ``data["cxcy"]`` holds the principal point, ``data["simple"]`` one focal."""

    def __init__(self, n: int | None, iters: int, device: str = "cpu"):
        self.n, self.iters, self.device = n, iters, device
        self._cache: dict = {}

    def __call__(self, pred: dict, data: dict) -> dict:
        _, _, h, w = data["image"].shape
        (cam_id,) = set(data["cam_id"])
        field = pred["rays"]
        key = (h, w, field.device)
        if key not in self._cache:
            idx = pixel_indices(self.n, h, w)
            x = np.ascontiguousarray(pixel_centers(idx, w, dtype=torch.float64).numpy())
            self._cache[key] = (idx.to(field.device), x)
        idx, x = self._cache[key]
        rays = field[:, idx]
        cxcy = data.get("cxcy")
        if isinstance(cxcy, Tensor):
            cxcy = cxcy.double().cpu().numpy()
        params = fit_rays(cam_id, rays if self.device.startswith("cuda") else rays.cpu().numpy(), x,
                          self.iters, cxcy=cxcy, simple=bool(data.get("simple", False)), device=self.device)
        return {"intrinsics": list(torch.from_numpy(params))}

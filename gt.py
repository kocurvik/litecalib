"""Ground-truth targets (radial knot angles, the FoV field at the grid) from AnyCalib camera models."""

import logging
from math import sqrt

import torch
from torch import Tensor

from litecalib import geometry
from litecalib.anycalib.cameras import CameraFactory

logger = logging.getLogger(__name__)


CENTER_TOL_PX = 1e-3
ASPECT_TOL = 1e-6


def check_center_and_aspect(
    cam_id: str,
    params: Tensor,
    h: int,
    w: int,
    center_tol: float = CENTER_TOL_PX,
    aspect_tol: float = ASPECT_TOL,
) -> Tensor:
    """Per-sample check that a camera has its principal point at the image center and square pixels."""
    cam = CameraFactory.create_from_id(cam_id)
    num_f = cam.NUM_F
    c = params[..., num_f : num_f + 2]
    centered = (c - c.new_tensor((0.5 * w, 0.5 * h))).abs().amax(dim=-1) <= center_tol
    if num_f == 1:
        return centered
    square = (params[..., 1] / params[..., 0] - 1).abs() <= aspect_tol
    return centered & square


def knot_pixels(
    h: int,
    w: int,
    k: int = geometry.DEFAULT_K,
    placement: str = geometry.DEFAULT_PLACEMENT,
    **kw,
) -> tuple[Tensor, Tensor]:
    """Knot pixel coordinates along the image diagonal, and their radii."""
    rho = geometry.knot_radii(h, w, k, placement, **kw)
    diag = rho.new_tensor((w, h)) / sqrt(float(h) ** 2 + float(w) ** 2)
    center = rho.new_tensor((0.5 * w, 0.5 * h))
    return center + rho[:, None] * diag, rho


def gt_knots(
    cam_id: str,
    params: Tensor,
    h: int,
    w: int,
    k: int = geometry.DEFAULT_K,
    placement: str = geometry.DEFAULT_PLACEMENT,
    check_invariants: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """GT knot angles for a batch of cameras sharing one input shape."""
    squeeze = params.ndim == 1
    params = params[None] if squeeze else params
    cam = CameraFactory.create_from_id(cam_id)

    if check_invariants:
        ok = check_center_and_aspect(cam_id, params, h, w)
        if not bool(ok.all()):
            logger.warning(
                "%d/%d GT cameras violate the centered-principal-point / square-pixel "
                "assumption at %dx%d",
                int((~ok).sum()),
                len(ok),
                h,
                w,
            )

    x, _ = knot_pixels(h, w, k, placement, dtype=params.dtype, device=params.device)
    x = x.expand(len(params), k, 2)
    bearings, _ = cam.unproject(params, x)

    theta = torch.atan2(torch.linalg.norm(bearings[..., :2], dim=-1), bearings[..., 2])

    parsed = cam.params_to_dict(params)
    _, valid_closed = geometry.psi(
        cam_id,
        torch.linalg.norm(x - parsed["c"].unsqueeze(-2), dim=-1),
        parsed["f"][..., 0],
        parsed["k"] if parsed["k"].shape[-1] else None,
    )
    valid = valid_closed & (theta > 0) & (theta < torch.pi)
    monotone = (torch.diff(theta, dim=-1) > 0).all(dim=-1)
    valid = valid.all(dim=-1) & monotone

    if squeeze:
        return theta[0], theta[0, -1], valid[0]
    return theta, theta[:, -1], valid


def normalized_radius(cam_id: str, params: Tensor, x: Tensor) -> Tensor:
    """``r = ||((x - cx) / fx, (y - cy) / fy)||`` of pixels ``x`` (..., N, 2). (..., N)"""
    parsed = CameraFactory.create_from_id(cam_id).params_to_dict(params)
    return torch.linalg.norm((x - parsed["c"].unsqueeze(-2)) / parsed["f"].unsqueeze(-2), dim=-1)


def gt_point_angles(cam_id: str, params: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """GT polar angles of the rays through pixels ``x`` (N, 2), for (B, D) cameras."""
    cam = CameraFactory.create_from_id(cam_id)
    x = x.reshape(1, -1, 2).expand(len(params), -1, -1)
    bearings, _ = cam.unproject(params, x)
    theta = torch.atan2(torch.linalg.norm(bearings[..., :2], dim=-1), bearings[..., 2])

    parsed = cam.params_to_dict(params)
    r = normalized_radius(cam_id, params, x)
    _, valid = geometry.psi(
        cam_id, r, torch.ones_like(r[..., 0]), parsed["k"] if parsed["k"].shape[-1] else None
    )
    valid = (valid & (theta >= 0) & (theta < torch.pi) & torch.isfinite(theta)).all(dim=-1)
    return theta, valid


def gt_point_fov(cam_id: str, params: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """GT FoV field ``(atan2(X, Z), atan2(Y, Z))`` of the rays through pixels ``x`` (N, 2)."""
    cam = CameraFactory.create_from_id(cam_id)
    bearings, _ = cam.unproject(params, x.reshape(1, -1, 2).expand(len(params), -1, -1))
    phi = torch.stack(
        (torch.atan2(bearings[..., 0], bearings[..., 2]), torch.atan2(bearings[..., 1], bearings[..., 2])),
        dim=-1,
    )
    _, valid = gt_point_angles(cam_id, params, x)
    return phi, valid & torch.isfinite(phi).flatten(-2).all(dim=-1)


def gt_field_knots(
    cam_id: str,
    params: Tensor,
    h: int,
    w: int,
    k: int = geometry.DEFAULT_GRID_K,
    field_repr: str = "fov",
) -> tuple[Tensor, Tensor]:
    """GT of the field variant for a batch of cameras of one shape."""
    squeeze = params.ndim == 1
    params = params[None] if squeeze else params
    x = geometry.grid_knot_pixels(h, w, k, dtype=params.dtype, device=params.device)
    assert field_repr in geometry.FIELD_REPRS, f"unknown field representation {field_repr!r}"
    phi, valid = gt_point_fov(cam_id, params, x)
    values = geometry.field_values(phi)
    if field_repr == "logmap":
        values = geometry.field_to_logmap(values)
    return (values[0], valid[0]) if squeeze else (values, valid)

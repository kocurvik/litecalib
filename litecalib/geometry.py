"""Geometry shared by the networks, the ground truth and the fits: knot layouts, the FoV field and the closed-form
radial angle profile ``theta = psi(rho)`` of each camera model."""

from functools import lru_cache
from math import pi, sqrt

import torch
from torch import Tensor

DEFAULT_K = 12
MAX_TAN_THETA = 0.5 * pi - 1e-3

KNOT_PLACEMENTS = ("u2", "u")
DEFAULT_PLACEMENT = "u2"

DOMAIN_TOL = 1e-6


def rho_max_of(h: int, w: int) -> float:
    """Half-diagonal of an ``h x w`` image, in pixels."""
    return 0.5 * sqrt(float(h) ** 2 + float(w) ** 2)


def knot_v(k: int = DEFAULT_K, **kw) -> Tensor:
    """Knot positions ``v_k = k / K`` in the placement variable, ``k = 1..K``. (K,)"""
    return torch.arange(1, k + 1, **{"dtype": torch.float32, **kw}) / k


def knot_u(k: int = DEFAULT_K, placement: str = DEFAULT_PLACEMENT, **kw) -> Tensor:
    """Normalized knot radii, ``u_K = 1``. (K,)"""
    assert placement in KNOT_PLACEMENTS, f"unknown knot placement {placement!r}"
    v = knot_v(k, **kw)
    return v.sqrt() if placement == "u2" else v


def knot_u_sq(k: int = DEFAULT_K, placement: str = DEFAULT_PLACEMENT, **kw) -> Tensor:
    """Squared normalized knot radii ``u_k^2``. (K,)"""
    return knot_u(k, placement, **kw) ** 2


def knot_radii(h: int, w: int, k: int = DEFAULT_K, placement: str = DEFAULT_PLACEMENT, **kw) -> Tensor:
    """Knot radii in pixels for an ``h x w`` input: ``rho_k = u_k * rho_max``. (K,)"""
    return knot_u(k, placement, **kw) * rho_max_of(h, w)


DEFAULT_GRID_K = 7


def grid_knot_v(k: int = DEFAULT_GRID_K, **kw) -> Tensor:
    """Positions along either grid side, ``v_j = -1 + 2 j / (K - 1)``, ``j = 0..K-1``. (K,)"""
    assert k >= 3 and k % 2 == 1, f"K must be odd and >= 3 (center + both edges), got {k}"
    return torch.linspace(-1.0, 1.0, k, **{"dtype": torch.float32, **kw})


def grid_num_knots(k: int) -> int:
    """Outputs of the grid variant: ``K^2``."""
    return k * k


def grid_knot_pixels(h: int, w: int, k: int = DEFAULT_GRID_K, **kw) -> Tensor:
    """Pixel coordinates of the ``K x K`` grid, row-major (``y`` outer, ``x`` inner). (K^2, 2)"""
    v = grid_knot_v(k, **kw)
    xs, ys = 0.5 * w * (1 + v), 0.5 * h * (1 + v)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(k * k, 2)


def grid_flip(values: Tensor, k: int, flip: int) -> Tensor:
    """(..., K^2) grid values of an image -> those of its horizontal (``flip == 1``) or vertical (``flip == 2``)
    mirror. ``flip == 0`` returns `values` unchanged.
    """
    if not flip:
        return values
    grid = values.reshape(*values.shape[:-1], k, k)
    return grid.flip(-1 if flip == 1 else -2).reshape(values.shape)


def field_num_values(k: int) -> int:
    """Outputs of the field variant: ``2 K^2``."""
    return 2 * k * k


def field_points(values: Tensor) -> Tensor:
    """(..., 2 K^2) field values -> (..., K^2, 2) ``(phi_x, phi_y)`` per grid point."""
    n = values.shape[-1] // 2
    return torch.stack((values[..., :n], values[..., n:]), dim=-1)


def field_values(points: Tensor) -> Tensor:
    """The inverse of `field_points`: (..., K^2, 2) -> (..., 2 K^2)."""
    return torch.cat((points[..., 0], points[..., 1]), dim=-1)


def field_flip(values: Tensor, k: int, flip: int) -> Tensor:
    """(..., 2 K^2) field values of an image -> those of its horizontal (``flip == 1``) or vertical (``flip ==
    2``) mirror: each component's grid mirrored as `grid_flip`, and the mirrored axis's component negated (a
    ray's ``X`` changes sign under a horizontal flip).
    """
    if not flip:
        return values
    n = k * k
    phi_x, phi_y = grid_flip(values[..., :n], k, flip), grid_flip(values[..., n:], k, flip)
    if flip == 1:
        phi_x = -phi_x
    else:
        phi_y = -phi_y
    return torch.cat((phi_x, phi_y), dim=-1)


FIELD_REPRS = ("fov", "logmap")


def rays_logmap(rays: Tensor) -> Tensor:
    """(..., 3) rays (any norm) -> (..., 2) log map ``theta (X, Y) / |(X, Y)|`` (0 on the axis)."""
    r = rays[..., :2].norm(dim=-1, keepdim=True)
    theta = torch.atan2(r, rays[..., 2:])
    return torch.where(r > 0, theta / r.clamp_min(torch.finfo(rays.dtype).tiny), 0.0) * rays[..., :2]


def logmap_rays(t: Tensor) -> Tensor:
    """The inverse of `rays_logmap`: (..., 2) -> (..., 3) unit rays ``(sin|t| t/|t|, cos|t|)``."""
    theta = t.norm(dim=-1, keepdim=True)
    sinc = torch.where(theta > 0, torch.sin(theta) / theta.clamp_min(torch.finfo(t.dtype).tiny), 1.0)
    return torch.cat((sinc * t, torch.cos(theta)), dim=-1)


def fov_rays(phi: Tensor) -> Tensor:
    """(..., 2) FoV field ``(atan2(X, Z), atan2(Y, Z))`` -> (..., 3) unit rays (``Z < 0`` past 90 deg: both
    cosines are negative there; `eval.evaluate.field_rays`' rule).
    """
    sx, cx, sy, cy = phi[..., 0].sin(), phi[..., 0].cos(), phi[..., 1].sin(), phi[..., 1].cos()
    rays = torch.stack((sx * cy, cx * sy, cx * cy), dim=-1)
    rays = rays * torch.where(cx + cy < 0, -1.0, 1.0).to(rays)[..., None]
    return rays / rays.norm(dim=-1, keepdim=True)


def rays_fov(rays: Tensor) -> Tensor:
    """(..., 3) rays -> (..., 2) FoV field ``(atan2(X, Z), atan2(Y, Z))``."""
    return torch.stack((torch.atan2(rays[..., 0], rays[..., 2]), torch.atan2(rays[..., 1], rays[..., 2])), -1)


def field_to_logmap(values: Tensor) -> Tensor:
    """(..., 2 K^2) FoV field values -> the same points' log map, same layout."""
    return field_values(rays_logmap(fov_rays(field_points(values))))


def field_from_logmap(values: Tensor) -> Tensor:
    """(..., 2 K^2) log-map values -> the same points' FoV field, same layout."""
    return field_values(rays_fov(logmap_rays(field_points(values))))


@lru_cache(maxsize=64)
def _quadrature_weights_cached(h: int, w: int, k: int, placement: str) -> Tensor:
    ys = torch.arange(h, dtype=torch.float64) + 0.5 - 0.5 * h
    xs = torch.arange(w, dtype=torch.float64) + 0.5 - 0.5 * w
    u_sq = (ys[:, None] ** 2 + xs[None, :] ** 2) / rho_max_of(h, w) ** 2
    v = u_sq if placement == "u2" else u_sq.sqrt()
    idx = torch.round(v * k).clamp_(1, k).long() - 1
    counts = torch.bincount(idx.reshape(-1), minlength=k)
    return (counts.to(torch.float64) / (h * w)).float()


def quadrature_weights(
    h: int, w: int, k: int = DEFAULT_K, placement: str = DEFAULT_PLACEMENT, **kw
) -> Tensor:
    """Per-knot weights that make the knot loss behave like a per-pixel loss."""
    assert placement in KNOT_PLACEMENTS, f"unknown knot placement {placement!r}"
    return _quadrature_weights_cached(h, w, k, placement).to(**kw)


MIN_INCREMENT = 1e-5


def profile_from_outputs(s: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
    """Map raw network outputs to a strictly increasing angle profile."""
    eps = torch.finfo(s.dtype).eps
    theta_max = pi * torch.sigmoid(s).clamp(eps, 1 - eps)
    p = torch.softmax(z, dim=-1)
    p = (p + MIN_INCREMENT) / (1 + z.shape[-1] * MIN_INCREMENT)
    phi = torch.cumsum(p, dim=-1)
    phi = torch.cat((phi[..., :-1], torch.ones_like(phi[..., -1:])), dim=-1)
    return theta_max.unsqueeze(-1) * phi, theta_max


def bias_init_for_profile(theta: Tensor) -> tuple[Tensor, Tensor]:
    """Head biases reproducing a target profile at initialization."""
    theta_max = theta[..., -1]
    s_bias = torch.logit(theta_max / pi)
    increments = torch.diff(theta, dim=-1, prepend=torch.zeros_like(theta[..., :1]))
    target = increments / theta_max.unsqueeze(-1)
    k = theta.shape[-1]
    p = (target * (1 + k * MIN_INCREMENT) - MIN_INCREMENT).clamp_min(1e-12)
    return s_bias, torch.log(p)


def parse_cam_id(cam_id: str) -> tuple[str, int]:
    """Split an AnyCalib camera id into (name, num_k), e.g. ``"radial:1" -> ("radial", 1)``."""
    name, _, num_k = cam_id.partition(":")
    if name.startswith("simple_"):
        name = name[len("simple_") :]
    defaults = {"pinhole": 0, "radial": 2, "kb": 4, "division": 1, "ucm": 1, "eucm": 2, "fov": 1}
    if name not in defaults:
        raise ValueError(f"Unsupported camera id: {cam_id!r}")
    return name, int(num_k) if num_k else defaults[name]


def psi_pinhole(r: Tensor, k: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``theta = atan(r)``. Always valid."""
    return torch.atan(r), torch.ones_like(r, dtype=torch.bool)


def _newton_tol(dtype: torch.dtype) -> float:
    """Convergence tolerance scaled to the working precision."""
    return 1e-5 * torch.finfo(dtype).eps / torch.finfo(torch.float32).eps


NEWTON_ITERS = 12


def _newton_poly(x: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """``(x (1 + k1 x^2 + ...) , d/dx of it)`` -- the polynomial and its slope at ``x``."""
    x_sq = x * x
    x_i = x_sq
    poly = 1 + k[..., :1] * x_sq
    grad = 1 + 3 * k[..., :1] * x_sq
    for j in range(1, k.shape[-1]):
        x_i = x_i * x_sq
        poly = poly + k[..., j, None] * x_i
        grad = grad + (3 + 2 * j) * k[..., j, None] * x_i
    return x * poly, grad


def _newton(
    r: Tensor, k: Tensor, x0: Tensor, iters: int, tol: float
) -> tuple[Tensor, Tensor]:
    """Solve ``x * (1 + k1 x^2 + k2 x^4 + ...) = r`` for ``x >= 0`` by Newton's method."""
    rd, kd = r.detach(), k.detach()
    x = x0.detach()
    can_stop = not torch._C._functorch.is_functorch_wrapped_tensor(x)
    for _ in range(iters):
        g, grad = _newton_poly(x, kd)
        step = (g - rd) / _nonzero(grad)
        x = x - step
        if can_stop and not bool((step.abs() > 1e-15 * x.abs().clamp_min(1.0)).any()):
            break
    g, _ = _newton_poly(x, kd)
    converged = (g - rd).abs() <= tol
    g_live, grad_live = _newton_poly(x, k)
    return x - (g_live - r) / _nonzero(grad_live), converged


def _nonzero(grad: Tensor) -> Tensor:
    """Keep a slope away from zero, so the Newton step stays finite at a fold."""
    return torch.where(grad.abs() < 1e-12, torch.full_like(grad, 1e-12), grad)


def _given_root(r: Tensor, k: Tensor, root: Tensor, tol: float) -> tuple[Tensor, Tensor]:
    """A root `newton_root` already solved for at these ``r, k``, and whether it is one."""
    g, _ = _newton_poly(root, k)
    return root, (g - r).abs() <= tol


def newton_root(cam_id: str, r: Tensor, k: Tensor | None = None) -> Tensor | None:
    """The root the Newton inverse of `cam_id` solves for at ``r`` -- ``t = tan(theta)`` for ``radial``,
    ``theta`` for ``kb`` -- or None for a model whose ``psi`` is closed-form.
    """
    name, _ = parse_cam_id(cam_id)
    if name not in ("radial", "kb"):
        return None
    x0 = r.clone() if name == "radial" else torch.sqrt(r.clamp_min(0))
    return _newton(r, k, x0, NEWTON_ITERS, _newton_tol(r.dtype))[0]


def psi_kb(
    r: Tensor, k: Tensor, iters: int = NEWTON_ITERS, tol: float | None = None,
    root: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Kannala-Brandt: solve ``r = theta + k1 theta^3 + k2 theta^5 + ...`` for ``theta``."""
    tol = _newton_tol(r.dtype) if tol is None else tol
    if root is None:
        theta, converged = _newton(r, k, torch.sqrt(r.clamp_min(0)), iters, tol)
    else:
        theta, converged = _given_root(r, k, root, tol)
    return theta, converged & (theta >= 0) & (theta < pi)


def brown_max_radius(k: Tensor) -> Tensor:
    """Largest undistorted radius ``t`` before the Brown-Conrady map folds back."""
    a = 3 * k[..., 0]
    b = 5 * k[..., 1] if k.shape[-1] > 1 else torch.zeros_like(a)
    discr = a * a - 4 * b
    den = torch.sqrt(discr.clamp_min(0)) - a
    finite = (discr >= 0) & (den > 0)
    safe_den = torch.where(finite, den, torch.ones_like(den))
    return torch.where(finite, torch.sqrt(2 / safe_den), torch.full_like(a, torch.inf))


def psi_brown(
    r: Tensor, k: Tensor, iters: int = NEWTON_ITERS, tol: float | None = None,
    root: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Brown-Conrady (radial only): solve ``r = t (1 + k1 t^2 + k2 t^4)``, ``theta = atan(t)``."""
    tol = _newton_tol(r.dtype) if tol is None else tol
    if root is None:
        t, converged = _newton(r, k, r.clone(), iters, tol)
    else:
        t, converged = _given_root(r, k, root, tol)
    t_max = brown_max_radius(k)
    unbounded = torch.isinf(t_max)
    t_sq = torch.where(unbounded, torch.ones_like(t_max), t_max) ** 2
    poly = 1 + k[..., 0] * t_sq
    if k.shape[-1] > 1:
        poly = poly + k[..., 1] * t_sq * t_sq
    r_max = torch.where(unbounded, torch.full_like(t_max, torch.inf), t_max * poly)
    fold = (r > r_max[..., None]) & (r <= r_max[..., None] * (1 + DOMAIN_TOL))
    t = torch.where(fold, t_max[..., None].expand_as(t), t)
    converged = converged | fold
    r = torch.where(fold, r_max[..., None].expand_as(r), r)
    valid = converged & (t >= 0) & (t <= t_max[..., None]) & (r <= r_max[..., None])
    return torch.atan(t), valid


def psi_division(r: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """Division model: ``theta = atan2(r, 1 + k1 r^2 + k2 r^4 + ...)``."""
    r_sq = r * r
    r_i = r_sq
    z = 1 + k[..., :1] * r_sq
    for i in range(1, k.shape[-1]):
        r_i = r_i * r_sq
        z = z + k[..., i, None] * r_i
    return torch.atan2(r, z), torch.ones_like(r, dtype=torch.bool)


def division_profile(u: Tensor, f_rel: Tensor, kappa: Tensor) -> Tensor:
    """The one-parameter division model's profile in half-diagonal units. (..., K)"""
    return torch.atan2(u / f_rel[..., None], 1 + kappa[..., None] * u * u)


def division_kappa(f: Tensor, k1: Tensor, rho_max: float | Tensor) -> Tensor:
    """AnyCalib's ``division:1`` ``(f, k1)`` (pixels) -> ``kappa = k1 (rho_max / f)^2``."""
    return k1 * (rho_max / f) ** 2


def division_k1(f: Tensor, kappa: Tensor, rho_max: float | Tensor) -> Tensor:
    """Inverse of `division_kappa`: ``k1 = kappa (f / rho_max)^2``."""
    return kappa * (f / rho_max) ** 2


def psi_ucm(r: Tensor, k: Tensor, tol: float = DOMAIN_TOL) -> tuple[Tensor, Tensor]:
    """UCM: ``eta = (xi + sqrt(1 + (1 - xi^2) r^2)) / (r^2 + 1)``, ``theta = atan2(eta r, eta - xi)``."""
    xi = k[..., :1]
    r_sq = r * r
    disc = 1 + (1 - xi * xi) * r_sq
    valid = disc >= -tol
    eta = (xi + torch.sqrt(disc.clamp_min(torch.finfo(r.dtype).eps))) / (r_sq + 1)
    return torch.atan2(eta * r, eta - xi), valid


def psi_eucm(r: Tensor, k: Tensor, tol: float = DOMAIN_TOL) -> tuple[Tensor, Tensor]:
    """EUCM: ``m_z = (1 - beta alpha^2 r^2) / (alpha sqrt(1 - (2 alpha - 1) beta r^2) + 1 - alpha)``."""
    alpha, beta = k[..., :1], k[..., 1:2]
    r_sq = r * r
    disc = 1 - (2 * alpha - 1) * beta * r_sq
    valid = disc >= -tol
    m_z = (1 - beta * alpha * alpha * r_sq) / (
        alpha * torch.sqrt(disc.clamp_min(torch.finfo(r.dtype).eps)) + (1 - alpha)
    )
    return torch.atan2(r, m_z), valid


def psi_fov(r: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """FOV model: ``theta = atan2(sin(r w), 2 tan(w/2) cos(r w))``."""
    w = k[..., :1]
    x = r * w
    valid = (x >= 0) & (x < pi) & (w > 0)
    return torch.atan2(torch.sin(x), 2 * torch.tan(0.5 * w) * torch.cos(x)), valid


_PSI_FNS = {
    "pinhole": psi_pinhole,
    "kb": psi_kb,
    "radial": psi_brown,
    "division": psi_division,
    "ucm": psi_ucm,
    "eucm": psi_eucm,
    "fov": psi_fov,
}


def psi(
    cam_id: str, rho: Tensor, f: Tensor, k: Tensor | None = None, root: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Closed-form radial angle profile of a camera model."""
    name, num_k = parse_cam_id(cam_id)
    if f.ndim == rho.ndim - 1:
        f = f.unsqueeze(-1)
    r = rho / f
    if name == "pinhole":
        return psi_pinhole(r)
    assert k is not None and k.shape[-1] == num_k, f"{cam_id} needs {num_k} coefficients"
    if root is not None:
        return _PSI_FNS[name](r, k, root=root)
    return _PSI_FNS[name](r, k)


def _dpsi_pinhole(r: Tensor, k: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``theta = atan(r)``."""
    return 1 / (1 + r * r), r.new_zeros(r.shape + (0,))


def _dpsi_brown(r: Tensor, k: Tensor, root: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``r = t (1 + k1 t^2 + ...)``, ``theta = atan(t)``: differentiate the inverse."""
    t = newton_root("radial", r, k) if root is None else root
    _, grad = _newton_poly(t, k)
    denom = (1 + t * t) * _nonzero(grad)
    powers = torch.stack([t ** (2 * j + 3) for j in range(k.shape[-1])], dim=-1)
    return 1 / denom, -powers / denom.unsqueeze(-1)


def _dpsi_kb(r: Tensor, k: Tensor, root: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``r = theta (1 + k1 theta^2 + ...)``: the same inverse, with theta itself as root."""
    theta = newton_root("kb", r, k) if root is None else root
    _, grad = _newton_poly(theta, k)
    denom = _nonzero(grad)
    powers = torch.stack([theta ** (2 * j + 3) for j in range(k.shape[-1])], dim=-1)
    return 1 / denom, -powers / denom.unsqueeze(-1)


def _dpsi_division(r: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """``theta = atan2(r, D)`` with ``D = 1 + k1 r^2 + ...``: ``dtheta = (D dr - r dD)/(r^2 + D^2)``."""
    r_sq = r * r
    d, dd_dr = torch.ones_like(r), torch.zeros_like(r)
    powers = []
    for j in range(k.shape[-1]):
        even = r_sq ** (j + 1)
        d = d + k[..., j, None] * even
        dd_dr = dd_dr + (2 * j + 2) * k[..., j, None] * r * r_sq**j
        powers.append(r * even)
    denom = r_sq + d * d
    return (d - r * dd_dr) / denom, -torch.stack(powers, dim=-1) / denom.unsqueeze(-1)


def _dpsi_ucm(r: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """``m = (xi + sqrt(1 + (1 - xi^2) r^2)) / (r^2 + 1)``, ``theta = atan2(m r, m - xi)``."""
    xi = k[..., :1]
    r_sq = r * r
    s = torch.sqrt((1 + (1 - xi * xi) * r_sq).clamp_min(torch.finfo(r.dtype).eps))
    den = r_sq + 1
    m = (xi + s) / den
    x, y = m * r, m - xi
    denom = _nonzero(x * x + y * y)
    dm_dr = ((1 - xi * xi) * r / s * den - (xi + s) * 2 * r) / (den * den)
    dm_dxi = (1 - xi * r_sq / s) / den
    dtheta_dr = (y * (m + r * dm_dr) - x * dm_dr) / denom
    dtheta_dxi = (y * r * dm_dxi - x * (dm_dxi - 1)) / denom
    return dtheta_dr, dtheta_dxi.unsqueeze(-1)


def _dpsi_eucm(r: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """``m = (1 - beta alpha^2 r^2) / (alpha sqrt(1 - (2 alpha - 1) beta r^2) + 1 - alpha)``, ``theta = atan2(r,
    m)``, so ``dtheta = (m dr - r dm) / (r^2 + m^2)``.
    """
    alpha, beta = k[..., :1], k[..., 1:2]
    r_sq = r * r
    s = torch.sqrt((1 - (2 * alpha - 1) * beta * r_sq).clamp_min(torch.finfo(r.dtype).eps))
    n = 1 - beta * alpha * alpha * r_sq
    den = alpha * s + 1 - alpha
    m = n / _nonzero(den)
    denom = _nonzero(r_sq + m * m)
    den_sq = _nonzero(den * den)

    ds_dr = -(2 * alpha - 1) * beta * r / s
    ds_da = -beta * r_sq / s
    ds_db = -(2 * alpha - 1) * r_sq / (2 * s)
    dn_dr, dn_da, dn_db = -2 * beta * alpha * alpha * r, -2 * beta * alpha * r_sq, -alpha * alpha * r_sq
    dden_dr, dden_da, dden_db = alpha * ds_dr, s + alpha * ds_da - 1, alpha * ds_db

    dm_dr = (dn_dr * den - n * dden_dr) / den_sq
    dm_da = (dn_da * den - n * dden_da) / den_sq
    dm_db = (dn_db * den - n * dden_db) / den_sq
    dtheta_dr = (m - r * dm_dr) / denom
    dk = torch.stack((-r * dm_da / denom, -r * dm_db / denom), dim=-1)
    return dtheta_dr, dk


_DPSI_FNS = {
    "pinhole": _dpsi_pinhole,
    "kb": _dpsi_kb,
    "radial": _dpsi_brown,
    "division": _dpsi_division,
    "ucm": _dpsi_ucm,
    "eucm": _dpsi_eucm,
}


def dpsi(
    cam_id: str, r: Tensor, k: Tensor | None = None, root: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Analytic derivatives of ``psi`` at normalized radii ``r = rho / f``."""
    name, num_k = parse_cam_id(cam_id)
    if name == "fov":
        raise NotImplementedError("the FOV model is ground-truth only; nothing fits it")
    if name == "pinhole":
        return _dpsi_pinhole(r)
    assert k is not None and k.shape[-1] == num_k, f"{cam_id} needs {num_k} coefficients"
    if root is not None:
        return _DPSI_FNS[name](r, k, root=root)
    return _DPSI_FNS[name](r, k)

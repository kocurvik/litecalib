"""Fitting intrinsics to a radial angle profile (used by the radial-profile ablation and the labels of the
direct-regression ablation): a linear initialization, then Levenberg-Marquardt on the angular residuals."""

import logging

import torch
from torch import Tensor

from litecalib import geometry
from litecalib.geometry import MAX_TAN_THETA, parse_cam_id

logger = logging.getLogger(__name__)


def _wls(a: Tensor, b: Tensor, w: Tensor) -> Tensor:
    """Weighted least squares ``argmin_x sum_k w_k (a_k.x - b_k)^2``."""
    aw = a * w.clamp_min(0).unsqueeze(-1)
    ata = aw.transpose(-1, -2) @ a
    atb = (aw.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    ridge = 1e-12 * torch.diagonal(ata, dim1=-2, dim2=-1).abs().amax(-1)[..., None, None]
    sol, info = torch.linalg.solve_ex(ata + ridge.clamp_min(1e-30) * eye, atb)
    if bool((info != 0).any()):
        logger.warning("singular weighted least-squares system for %d sample(s)", int((info != 0).sum()))
        sol = torch.where((info == 0)[..., None], sol, torch.zeros_like(sol))
    return sol


def _tan_rows(theta: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """``tan(theta)`` with knots at/past 90 deg removed (weight set to 0)."""
    usable = theta < MAX_TAN_THETA
    if not bool(usable.all()):
        logger.warning(
            "dropping %d knot(s) with theta >= 90 deg from a tan-based linear fit",
            int((~usable).sum()),
        )
    return torch.tan(theta.clamp(max=MAX_TAN_THETA)), w * usable


def _odd_poly_fit(x: Tensor, target: Tensor, w: Tensor, n_terms: int) -> Tensor:
    """Weighted fit of ``target = a_0 x + a_1 x^3 + ... + a_{n-1} x^(2n-1)``."""
    powers = torch.arange(1, 2 * n_terms, 2, device=x.device, dtype=x.dtype)
    x_scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)
    sol = _wls((x / x_scale).unsqueeze(-1) ** powers, target, w)
    return sol / x_scale**powers


def linear_init(
    cam_id: str,
    rho: Tensor,
    theta: Tensor,
    w: Tensor,
    eucm_init: str = "proxy",
) -> tuple[Tensor, Tensor]:
    """Linear (weighted least-squares) initialization of the intrinsics."""
    name, num_k = parse_cam_id(cam_id)
    if name == "fov":
        raise ValueError("the FOV model is ground-truth only and cannot be fitted")
    in_dtype = rho.dtype
    rho, theta, w = rho.double(), theta.double(), w.double()
    scale = rho.amax(dim=-1, keepdim=True).clamp_min(torch.finfo(rho.dtype).eps)
    rho_n = rho / scale

    if name == "pinhole":
        t, w = _tan_rows(theta, w)
        f = _wls(t.unsqueeze(-1), rho_n, w)
        k = rho.new_zeros(len(rho), 0)

    elif name == "kb":
        sol = _odd_poly_fit(theta, rho_n, w, num_k + 1)
        f, k = sol[..., :1], sol[..., 1:] / sol[..., :1]

    elif name == "radial":
        t, w = _tan_rows(theta, w)
        sol = _odd_poly_fit(t, rho_n, w, num_k + 1)
        f, k = sol[..., :1], sol[..., 1:] / sol[..., :1]

    elif name == "division":
        t, w = _tan_rows(theta, w)
        powers = torch.arange(2, 2 * num_k + 2, 2, device=rho.device, dtype=rho.dtype)
        design = torch.cat(
            (t.unsqueeze(-1), t.unsqueeze(-1) * rho_n.unsqueeze(-1) ** powers), dim=-1
        )
        sol = _wls(design, rho_n, w)
        f = sol[..., :1]
        exps = torch.arange(1, 2 * num_k, 2, device=rho.device, dtype=rho.dtype)
        k = sol[..., 1:] * f**exps

    elif name == "ucm":
        s, c = torch.sin(theta), torch.cos(theta)
        design = torch.stack((s, -rho_n), dim=-1)
        sol = _wls(design, rho_n * c, w)
        f, k = sol[..., :1], sol[..., 1:]
        f0 = _wls(s.unsqueeze(-1), rho_n * c, w)
        bad = k[..., :1] < 0
        f = torch.where(bad, f0, f)
        k = torch.where(bad, torch.zeros_like(k), k)

    elif name == "eucm" and eucm_init == "ucm":
        f_u, xi = linear_init("ucm", rho, theta, w)
        f = f_u.unsqueeze(-1) / scale / (1 + xi)
        k = torch.cat((xi / (1 + xi), torch.ones_like(xi)), dim=-1)

    elif name == "eucm":
        f_proxy, _ = linear_init("kb:3", rho, theta, w)
        r = rho / f_proxy.unsqueeze(-1)
        R, Z = torch.sin(theta), torch.cos(theta)
        zr_r = Z * r - R
        sol = _wls(torch.stack(((R * r) ** 2, 2 * Z * r * zr_r), dim=-1), zr_r**2, w)
        alpha = sol[..., 1:2].clamp(1e-4, 1 - 1e-4)
        beta = torch.where(alpha < 0.025, torch.ones_like(alpha), sol[..., :1] / alpha**2)
        f = (f_proxy.unsqueeze(-1) / scale)
        k = torch.cat((alpha, beta.clamp(1e-4, 1e2)), dim=-1)

    else:
        raise ValueError(f"Unsupported camera id: {cam_id!r}")

    return (f * scale).squeeze(-1).to(in_dtype), k.to(in_dtype)


_SOFTPLUS_LIMIT = 20.0

DEFAULT_GN_ITERS = {"eucm": 10}
DEFAULT_GN_ITERS_OTHER = 2


def _pack(name: str, f: Tensor, k: Tensor) -> Tensor:
    """Intrinsics -> unconstrained optimization variables."""
    eta = [torch.log(f.clamp_min(1e-6)).unsqueeze(-1)]
    if name == "ucm":
        eta.append(_inv_softplus(k[..., :1]))
    elif name == "eucm":
        eta += [torch.logit(k[..., :1].clamp(1e-6, 1 - 1e-6)), _inv_softplus(k[..., 1:2])]
    elif k.shape[-1]:
        eta.append(k)
    return torch.cat(eta, dim=-1)


def _unpack(name: str, eta: Tensor) -> tuple[Tensor, Tensor]:
    """Unconstrained optimization variables -> intrinsics."""
    f = torch.exp(eta[..., 0])
    rest = eta[..., 1:]
    if name == "ucm":
        return f, torch.nn.functional.softplus(rest)
    if name == "eucm":
        return f, torch.cat(
            (torch.sigmoid(rest[..., :1]), torch.nn.functional.softplus(rest[..., 1:2])),
            dim=-1,
        )
    return f, rest


def _inv_softplus(x: Tensor) -> Tensor:
    x = x.clamp_min(1e-6)
    return torch.where(x > _SOFTPLUS_LIMIT, x, torch.log(torch.expm1(x.clamp_max(_SOFTPLUS_LIMIT))))


def _dparams_deta(name: str, eta: Tensor) -> Tensor:
    """(..., num_k) derivative of each coefficient w.r.t. its optimization variable."""
    rest = eta[..., 1:]
    if name == "ucm":
        return torch.sigmoid(rest)
    if name == "eucm":
        alpha = torch.sigmoid(rest[..., :1])
        return torch.cat((alpha * (1 - alpha), torch.sigmoid(rest[..., 1:2])), dim=-1)
    return torch.ones_like(rest)


LM_RTOL = 1e-6
_LM_LAMBDA_MAX = 1e8


def _levenberg_marquardt(
    eta: Tensor, residual, jacobian, w: Tensor, iters: int, damping: float, retries: int,
    rtol: float = LM_RTOL,
) -> Tensor:
    """Minimize ``sum_k w_k residual_k(eta)^2`` per sample; the loop `refine` describes."""
    res, aux = residual(eta)
    cost = (w * res**2).sum(dim=-1)
    lam = eta.new_full(eta.shape[:-1], damping)
    eye = torch.eye(eta.shape[-1], device=eta.device, dtype=eta.dtype)
    active = torch.isfinite(cost)
    for _ in range(iters):
        if not bool(active.any()):
            break
        jac = jacobian(eta, aux)
        jw = (jac * w.unsqueeze(-1)).transpose(-1, -2)
        jtj = jw @ jac
        jtr = (jw @ res.unsqueeze(-1)).squeeze(-1)
        diag = torch.diagonal(jtj, dim1=-2, dim2=-1).clamp_min(1e-12)

        pending = active.clone()
        for _ in range(retries):
            damped = jtj + (lam[..., None] * diag).diag_embed() + 1e-12 * eye
            delta, info = torch.linalg.solve_ex(damped, jtr)
            eta_new = torch.nan_to_num(eta - delta, nan=0.0, posinf=0.0, neginf=0.0)
            eta_new = torch.where((info == 0)[..., None], eta_new, eta)
            res_new, aux_new = residual(eta_new)
            cost_new = (w * res_new**2).sum(dim=-1)
            good = pending & (cost_new < cost)
            active = active & ~(good & (cost - cost_new <= rtol * cost))
            eta = torch.where(good[..., None], eta_new, eta)
            res = torch.where(good[..., None], res_new, res)
            if aux is not None:
                aux = torch.where(good.view(-1, *([1] * (aux.ndim - 1))), aux_new, aux)
            cost = torch.where(good, cost_new, cost)
            lam = torch.where(good, lam * 0.1, lam)
            lam = torch.where(pending & ~good, lam * 10, lam).clamp(1e-12, _LM_LAMBDA_MAX)
            pending = pending & ~good
            if not bool(pending.any()):
                break
        active = active & ~(pending & (lam >= _LM_LAMBDA_MAX))
    return eta


def refine(
    cam_id: str,
    f: Tensor,
    k: Tensor,
    rho: Tensor,
    theta: Tensor,
    w: Tensor,
    iters: int = 2,
    damping: float = 1e-4,
    retries: int = 4,
) -> tuple[Tensor, Tensor]:
    """Refinement of ``sum_k w_k (theta_k - psi_M(rho_k; eta))^2``."""
    name, _ = parse_cam_id(cam_id)
    in_dtype = f.dtype
    f, k, rho, theta, w = (t.double() for t in (f, k, rho, theta, w))
    eta = _pack(name, f, k)
    scale = rho.amax(dim=-1).clamp_min(torch.finfo(rho.dtype).eps)
    rho_n, w = rho / scale.unsqueeze(-1), w.clamp_min(0)

    def radii(eta_: Tensor) -> tuple[Tensor, Tensor]:
        """(B, K) normalized radii ``r = rho / f`` and (B, num_k) coefficients at `eta_`."""
        f_, k_ = _unpack(name, eta_)
        return rho_n / (f_ / scale).unsqueeze(-1), k_

    def residual(eta_: Tensor) -> tuple[Tensor, Tensor | None]:
        """(B, K) angular residuals, and the Newton root the Jacobian reuses (or None)."""
        r, k_ = radii(eta_)
        k_ = k_ if k_.shape[-1] else None
        root = geometry.newton_root(cam_id, r, k_)
        pred, _ = geometry.psi(cam_id, r, torch.ones_like(scale), k_, root=root)
        return theta - pred, root

    def jacobian_analytic(eta_: Tensor, root: Tensor | None) -> Tensor:
        """(B, K, P): d(residual)/d(eta), in closed form, at the point `root` was solved at."""
        r, k_ = radii(eta_)
        dr, dk = geometry.dpsi(cam_id, r, k_ if k_.shape[-1] else None, root=root)
        cols = [(r * dr).unsqueeze(-1)]
        if k_.shape[-1]:
            cols.append(-dk * _dparams_deta(name, eta_).unsqueeze(-2))
        return torch.cat(cols, dim=-1)

    eta = _levenberg_marquardt(eta, residual, jacobian_analytic, w, iters, damping, retries)

    f, k = _unpack(name, eta)
    return f.to(in_dtype), k.to(in_dtype)


def params_to_anycalib(
    cam_id: str, f: Tensor, k: Tensor, h: int, w: int, center: Tensor | None = None
) -> Tensor:
    """Assemble AnyCalib-format intrinsics with the principal point at the image center."""
    simple = cam_id.startswith("simple_")
    focals = f.unsqueeze(-1) if simple else f.unsqueeze(-1).expand(-1, 2)
    if center is None:
        center = f.new_tensor((0.5 * w, 0.5 * h))
    center = center.to(f.dtype).expand(len(f), 2)
    return torch.cat((focals, center, k), dim=-1)


def fit_profile(
    cam_id: str,
    rho: Tensor,
    theta: Tensor,
    b: Tensor | None = None,
    h: int | None = None,
    w: int | None = None,
    gn_iters: int | None = None,
    eucm_init: str = "proxy",
    center: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Fit intrinsics to a predicted radial angle profile."""
    assert rho.ndim == theta.ndim == 2, f"expected (B, K) inputs, got {rho.shape=}"
    if gn_iters is None:
        gn_iters = DEFAULT_GN_ITERS.get(parse_cam_id(cam_id)[0], DEFAULT_GN_ITERS_OTHER)
    weights = torch.ones_like(rho) if b is None else 1 / b.clamp_min(1e-6) ** 2
    f, k = linear_init(cam_id, rho, theta, weights, eucm_init=eucm_init)
    f, k = refine(cam_id, f, k, rho, theta, weights, iters=gn_iters)
    if h is None or w is None:
        return f, k
    return params_to_anycalib(cam_id, f, k, h, w, center)


@torch.no_grad()
def division_labels(
    theta: Tensor, h: int, w: int, placement: str = geometry.DEFAULT_PLACEMENT,
    cam_id: str = "division:1",
) -> tuple[Tensor, Tensor]:
    """Parameter-regression labels of the (f, k1) network: ``division:1`` fitted to GT knots."""
    rho_max = geometry.rho_max_of(h, w)
    k = theta.shape[-1]
    rho = geometry.knot_radii(h, w, k, placement, dtype=torch.float64, device=theta.device)
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        f, k1 = fit_profile(cam_id, rho.expand(len(theta), k), theta.double())
    finally:
        logger.setLevel(level)
    kappa = geometry.division_kappa(f, k1[..., 0], rho_max) if k1.shape[-1] else torch.zeros_like(f)
    return torch.log(f / rho_max).to(theta.dtype), kappa.to(theta.dtype)


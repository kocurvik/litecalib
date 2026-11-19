"""Training losses: the quadrature-weighted L1 / Laplace-NLL on the network's outputs, and the parameter loss
of the direct-regression ablation."""

import math

import torch
from torch import Tensor

RAD2DEG = 180 / math.pi
LOGB_RANGE = (math.log(1e-4), math.log(1.0))


def profile_loss(
    pred: dict[str, Tensor],
    theta_gt: Tensor,
    omega: Tensor,
    valid: Tensor | None = None,
    nll: bool = True,
    lambda_theta: float = 0.0,
    theta_max_gt: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Weighted Laplace-NLL (or weighted L1) over the outputs, averaged over the valid samples."""
    theta = pred["theta"]
    assert theta.shape == theta_gt.shape, f"{theta.shape=} != {theta_gt.shape=}"
    keep = torch.ones(len(theta), dtype=torch.bool, device=theta.device) if valid is None else valid
    n = int(keep.sum())
    if n == 0:
        zero = (0.0 * theta.sum()) + (0.0 * pred["Theta"].sum()) + (0.0 * pred["logb"].sum())
        return zero, {"n": 0, "l1_deg": float("nan"), "fov_err_deg": float("nan"), "b_deg": float("nan")}

    w = keep.to(theta.dtype)
    abs_err = (theta - theta_gt).abs()
    l1 = (omega * abs_err).sum(dim=-1)
    theta_max_gt = theta_gt[:, -1] if theta_max_gt is None else theta_max_gt

    logb = pred["logb"].clamp(*LOGB_RANGE)
    b = torch.exp(logb)
    if nll:
        per_sample = (omega * (abs_err / b + logb)).sum(dim=-1)
    else:
        per_sample = l1 + 0.0 * logb.sum(dim=-1)

    fov_err = (pred["Theta"] - theta_max_gt).abs()
    if lambda_theta:
        per_sample = per_sample + lambda_theta * fov_err

    loss = (w * per_sample).sum() / n
    with torch.no_grad():
        kept = keep.nonzero(as_tuple=True)[0]
        metrics = {
            "n": n,
            "l1_deg": float(RAD2DEG * l1[kept].mean()),
            "fov_err_deg": float(RAD2DEG * 2 * fov_err[kept].median()),
            "b_deg": float(RAD2DEG * b[kept].mean()),
        }
    return loss, metrics


def division_param_loss(
    pred: dict[str, Tensor],
    log_f_gt: Tensor,
    kappa_gt: Tensor,
    valid: Tensor | None = None,
    kappa_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Parameter regression for the (f, k1) network: ``|d log f| + kappa_weight |d kappa|``."""
    log_f, kappa = pred["log_f"], pred["kappa"]
    keep = torch.ones(len(log_f), dtype=torch.bool, device=log_f.device) if valid is None else valid
    n = int(keep.sum())
    if n == 0:
        return 0.0 * (log_f.sum() + kappa.sum()), {"f_err_pct": float("nan"), "kappa_err": float("nan")}
    err_f = (log_f[keep] - log_f_gt[keep]).abs()
    err_k = (kappa[keep] - kappa_gt[keep]).abs()
    loss = (err_f + kappa_weight * err_k).mean()
    return loss, {
        "f_err_pct": float(100 * err_f.detach().mean()),
        "kappa_err": float(err_k.detach().mean()),
    }

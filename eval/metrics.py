"""AnyCalib's evaluation metrics (siclib's `compute_metrics` and `run_eval` aggregation), with the ground-truth
half computed once per image, and the camera model each benchmark is fitted with."""

import math

import numpy as np
import torch

from litecalib.anycalib.cameras import CameraFactory
from litecalib.anycalib.manifolds import Unit3

RAD2DEG = 180 / math.pi

EVAL_CAM = {
    "stanford2d3d": "pinhole",
    "tartanair": "pinhole",
    "megadepth2k": "pinhole",
    "lamar2k": "pinhole",
    "stanford2d3d_edit": "pinhole",
    "tartanair_edit": "pinhole",
    "lamar2k_edit": "pinhole",
    "megadepth2k-radial": "radial:1",
    "scannetpp2k": "kb:4",
    "monovo2k": "ucm",
    "influx_val": "gt",
    "influx_pp_val": "gt",
    "cathedral": "gt",
    "openpano_v2": "pinhole",
    "openpano_v2_gen": "gt",
}
FIT_GT = "gt"
THRESHOLDS = [1, 2, 5, 10]
REPROJ_THRESHOLDS = [10, 20, 50, 100]
PIXEL_THRESHOLDS = [1, 3, 5]
SUMMARY_KEYS = ("ang_error", "reproj_error", "vfov_error", "hfov_error", "vfov_pin_error", "max_rel_f_error")
AUC_KEYS = {"ang_error": THRESHOLDS, "vfov_error": THRESHOLDS, "hfov_error": THRESHOLDS,
            "reproj_error": REPROJ_THRESHOLDS}


def compute_auc(errors: np.ndarray, thresholds: list[float], min_error: float | None = None) -> list[float]:
    """Area under the recall-error curve up to each threshold, normalized (siclib's `compute_auc`)."""
    errors = np.sort(np.where(np.isnan(errors), np.inf, errors))
    recall = (np.arange(len(errors)) + 1) / len(errors)
    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round(trapz(r, x=e) / t, 4))
    return aucs


def summarize(per_image: list[dict], keys: tuple[str, ...] = SUMMARY_KEYS) -> dict:
    """Mean, median and AUCs of each metric over a benchmark, as siclib's SimplePipeline.run_eval."""
    out: dict[str, float] = {"n": len(per_image)}
    for k in keys:
        arr = np.array([r[k] for r in per_image if k in r], dtype=float)
        if not len(arr):
            continue
        out[f"mean_{k}"] = float(np.nanmean(arr))
        out[f"median_{k}"] = float(np.nanmedian(arr))
        if k in AUC_KEYS:
            out[f"nans%_{k}"] = float(100 * np.isnan(arr).mean())
            for t, a in zip(AUC_KEYS[k], compute_auc(arr, list(AUC_KEYS[k]), min_error=1)):
                out[f"auc_{k}@{t}"] = float(a)
    return out


class GroundTruth:
    """The GT-only half of `compute_metrics` for one image and one fitted camera model."""

    def __init__(self, cam, cam_id_gt: str, intrinsics_gt: torch.Tensor, w: int, h: int):
        self.cam_gt = cam_gt = CameraFactory.create_from_id(cam_id_gt)
        self.intrins_gt = intrins_gt = intrinsics_gt[: cam_gt.nparams]
        self.w, self.h = w, h
        vfov_gt, self.valid_vfov_gt = cam_gt.get_vfov(intrins_gt, h)
        self.vfov_gt = vfov_gt * RAD2DEG
        hfov_gt, self.valid_hfov_gt = cam_gt.get_hfov(intrins_gt, w)
        self.hfov_gt = hfov_gt * RAD2DEG
        if getattr(cam_gt, "NUM_F", 0) > 0:
            self.vfov_pin_gt = cam_gt.get_pinhole_vfov(intrins_gt, h) * RAD2DEG
        self.grid = cam.pixel_grid_coords(h, w, intrins_gt, offset=0.5)
        self.bearings_gt, self.valid_gt = cam_gt.unproject(intrins_gt, self.grid)
        if "radial" in cam.id and "radial" in cam_gt.id:
            k1_idx_gt = cam_gt.PARAMS_IDX["k1"]
            cam_pin = CameraFactory.create_from_id("pinhole")
            self.bearings_pin, _ = cam_pin.unproject(intrins_gt[:k1_idx_gt], self.grid)
            self.im_gt, self.v1 = cam_gt.project(intrins_gt, self.bearings_pin)


def metrics(pred_intrinsics: torch.Tensor, gt: GroundTruth, cam, scale_xy, shift_xy,
            thresholds: list[float], cxcy=None) -> dict:
    """`compute_metrics` for one predicted camera, with `gt`'s half precomputed."""
    intrins = cam.reverse_scale_and_shift(pred_intrinsics, scale_xy, shift_xy)
    cam_gt, intrins_gt = gt.cam_gt, gt.intrins_gt
    results = {}
    w_cxcy_prior = cxcy is not None
    w, h = gt.w, gt.h
    if cam.id == cam_gt.id:
        abs_errors = torch.abs(intrins - intrins_gt)
        rel_errors = abs_errors / intrins_gt.abs()
        for abs_e, rel_e, p_str in zip(abs_errors, rel_errors, cam.PARAMS_IDX.keys()):
            if p_str in ("cx", "cy") and w_cxcy_prior:
                continue
            results[f"{p_str}_error"] = abs_e.item()
            results[f"rel_{p_str}_error"] = rel_e.item()
        if "fx" in cam.PARAMS_IDX:
            rel_fx = rel_errors[cam.PARAMS_IDX["fx"]]
            rel_fy = rel_errors[cam.PARAMS_IDX["fy"]]
            results["max_rel_f_error"] = max(rel_fx, rel_fy).item()
        abs_cx = abs_errors[cam.PARAMS_IDX["cx"]]
        abs_cy = abs_errors[cam.PARAMS_IDX["cy"]]
        results["max_rel_c_error"] = 2 * max(abs_cx / w, abs_cy / h).item()
    vfov, valid_vfov = cam.get_vfov(intrins, h)
    vfov = vfov * RAD2DEG
    results["vfov_gt"] = gt.vfov_gt.item()
    results["vfov"] = vfov.item()
    if valid_vfov and gt.valid_vfov_gt:
        results["vfov_error"] = torch.abs(vfov - gt.vfov_gt).item()
    else:
        results["vfov_error"] = float("nan")
    hfov, valid_hfov = cam.get_hfov(intrins, w)
    hfov = hfov * RAD2DEG
    results["hfov_gt"] = gt.hfov_gt.item()
    results["hfov"] = hfov.item()
    if valid_hfov and gt.valid_hfov_gt:
        results["hfov_error"] = torch.abs(hfov - gt.hfov_gt).item()
    else:
        results["hfov_error"] = float("nan")
    if getattr(cam, "NUM_F", 0) > 0 and getattr(cam_gt, "NUM_F", 0) > 0:
        vfov_pin = cam.get_pinhole_vfov(intrins, h) * RAD2DEG
        results["vfov_pin_error"] = torch.abs(vfov_pin - gt.vfov_pin_gt).item()
    grid = gt.grid
    im_coords_pred, valid_p = cam.project(intrins, gt.bearings_gt)
    errors = torch.linalg.norm(grid - im_coords_pred, dim=-1)
    valid_u = gt.valid_gt
    if valid_u is not None and valid_p is not None:
        errors = errors[valid_u & valid_p]
    elif valid_u is not None:
        errors = errors[valid_u]
    elif valid_p is not None:
        errors = errors[valid_p]
    results["reproj_error"] = errors.mean().item()
    for th in thresholds:
        results[f"reproj_error@{th}"] = (errors < th).float().mean().item()
    bearings, valid = cam.unproject(intrins, grid)
    ang_errors = RAD2DEG * Unit3.distance(gt.bearings_gt, bearings)
    valid_gt = gt.valid_gt
    if valid_gt is not None and valid is not None:
        ang_errors = ang_errors[valid & valid_gt]
    elif valid is not None:
        ang_errors = ang_errors[valid]
    elif valid_gt is not None:
        ang_errors = ang_errors[valid_gt]
    results["ang_error"] = ang_errors.mean().item()
    results["ang_med_error"] = ang_errors.median().item()
    if "radial" in cam.id and "radial" in cam_gt.id:
        k1_idx_gt = cam_gt.PARAMS_IDX["k1"]
        k1_idx = cam.PARAMS_IDX["k1"]
        im_pred, v2 = CameraFactory.create_from_id(f"{cam_gt.id[:-1]}{cam.num_k}").project(
            torch.cat((intrins_gt[:k1_idx_gt], intrins[k1_idx:])), gt.bearings_pin
        )
        v1 = gt.v1
        assert v1 is not None and v2 is not None
        err = torch.linalg.norm(gt.im_gt - im_pred, dim=-1)[v1 & v2]
        for th in thresholds:
            results[f"pixel_distortion_error@{th}"] = (err < th).float().mean().item()
        if cam.id == cam_gt.id:
            J = cam_gt.jac_bearings_wrt_params(
                intrins_gt, bearings.reshape(-1, 3), grid.reshape(-1, 2)
            ).flatten(0, 1)
            inv_cov = J.transpose(-1, -2) @ J
            err = intrins - intrins_gt
            results["maha_error"] = (err[None] @ inv_cov @ err[:, None]).item()
    return results

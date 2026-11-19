"""Evaluate LiteCalib checkpoints and AnyCalib models on the same benchmarks, through one path: the same images,
the same ground truth, the same fitted camera model per benchmark and AnyCalib's own metrics.

    # a LiteCalib checkpoint (c and a held, AnyCalib's calibrator compiled, 2 iterations: the paper's fit)
    python eval/evaluate.py --model checkpoints/vitl14_..._ms_aux_field_wsd24_k5.pt --data stanford2d3d monovo2k

    # AnyCalib's released model, or AnyCalib's network trained here (an `arch: anycalib` checkpoint)
    python eval/evaluate.py --model anycalib_gen --data stanford2d3d

    # the sparse-field study: AnyCalib's calibrator on N x N pixels of its dense field
    python eval/evaluate.py --model anycalib_gen --anycalib_sampling dense egrid:5 acnb_egrid:5:it2 --data ...

A dataset is a benchmark directory under the `benchmarks` root (`<name>/images.{h5,csv}`) or `name:split` for a
generated view dataset (`openpano_v2_gen:test`).
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as data_lib
import metrics as metrics_lib
import paths
import slurm
import sparse
from litecalib import fit, geometry
from litecalib.anycalib.cameras import CameraFactory
from litecalib.model import REGRESSION_CAMS
from litecalib.predictor import (build_network, crop_resize_plan, field_rays, fit_rays, load_checkpoint,
                                 target_size)

SPLITS = ("train", "val", "test")
OURS_FIT_DEFAULT = "acsquare_it2"
_OURS_FIT = re.compile(r"^(?:ac(free|held|square)_it(\d+)|it(\d+))$")


def parse_spec(spec: str) -> tuple[str, str | None]:
    """`openpano_v2_gen:test` -> ('openpano_v2_gen', 'test'); `monovo2k` -> ('monovo2k', None)."""
    head, sep, tail = spec.rpartition(":")
    return (head, tail) if sep and tail in SPLITS else (spec, None)


def load_datapoints(spec: str, root: Path | None, machine: str | None) -> tuple[str, list]:
    """Resolve a dataset spec to (label, datapoints)."""
    name, split = parse_spec(spec)
    if split is not None:
        return spec, data_lib.load_index(data_lib.resolve_dataset_dir(name, machine), split)
    directory = Path(name) if Path(name).exists() else (root or paths.get("benchmarks", machine)) / name
    return spec, data_lib.load_index(directory)


def group_by_shape(datapoints: list, limit: int | None = None) -> list[dict]:
    """Batchable groups: same image size and same ground-truth camera model."""
    groups: dict[tuple, dict] = {}
    for dp in datapoints[:limit]:
        key = (str(dp.cam_id), int(dp.h), int(dp.w))
        entry = groups.setdefault(
            key, {"cam_id": key[0], "h": key[1], "w": key[2], "names": [], "files": [], "params": []}
        )
        entry["names"].append(dp.name)
        entry["files"].append(dp)
        entry["params"].append(np.asarray(dp.params, dtype=np.float64))
    for entry in groups.values():
        entry["params"] = torch.from_numpy(np.stack(entry["params"]))
    return list(groups.values())


def _sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class Timed:
    """Keeps the per-image cost of a predictor's fit (the first call, which compiles, is not counted)."""

    def _init_timing(self) -> None:
        self.fit_seconds, self.fit_images, self.warm = 0.0, 0, False

    def _timed(self, fn, n: int, device):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        if self.warm:
            self.fit_seconds += time.perf_counter() - t0
            self.fit_images += n
        self.warm = True
        return out


class OursPredictor(Timed):
    """A LiteCalib checkpoint and its fit.

    Field checkpoints are fitted by AnyCalib's calibrator (compiled) on their K^2 rays: ``acsquare_it<N>``
    (c at the image center, one focal: the paper's fit), ``acheld_it<N>`` (c held, fx, fy free) or
    ``acfree_it<N>`` (c and fx, fy from AnyCalib's linear fit). Radial checkpoints are fitted by the profile fit
    with ``it<N>`` iterations; a division checkpoint is its own ``division:1`` camera.
    """

    def __init__(self, ckpt: Path, device: str, fit_mode: str = OURS_FIT_DEFAULT, fit_device: str = "cpu"):
        config, state = load_checkpoint(ckpt)
        self.net = build_network(config, state).eval().to(device)
        self.profile = self.net.profile
        m = _OURS_FIT.match(fit_mode)
        if m is None:
            raise ValueError(f"unknown fit {fit_mode!r}: ac{{free,held,square}}_it<N> or it<N>")
        self.variant = m.group(1)
        self.iters = int(m.group(2) or m.group(3))
        if (self.variant is None) != (self.profile != "field"):
            raise ValueError(f"fit {fit_mode!r} does not apply to a {self.profile} checkpoint")
        self.device, self.fit_device = device, fit_device
        self.div = self.net.backbone.input_divisor
        self.name = Path(ckpt).stem + f"@{fit_mode}" + ("" if fit_device == "cpu" else f"@{fit_device}")
        self.info = {"type": "litecalib", "checkpoint": str(ckpt), "config": config, "fit": fit_mode,
                     "fit_device": fit_device}
        self._init_timing()

    @torch.no_grad()
    def __call__(self, images: list[Tensor], cam_id: str, h: int, w: int) -> Tensor:
        ht, wt = target_size(h, w, self.div)
        (top, left, crop_h, crop_w), (sx, sy) = crop_resize_plan(h, w, ht, wt)
        batch = torch.stack(images).to(self.device)[..., top : top + crop_h, left : left + crop_w]
        if (crop_h, crop_w) != (ht, wt):
            batch = torch.nn.functional.interpolate(batch, (ht, wt), mode="bicubic", align_corners=False,
                                                    antialias=sy < 1).clamp(0, 1)
        out = self.net(batch)
        shift = (-left * sx, -top * sy)
        if self.profile == "field":
            return self._timed(lambda: self._fit_field(out, cam_id, ht, wt, (sx, sy), shift), len(images),
                               self.fit_device)
        return self._timed(lambda: self._fit_knots(out, cam_id, ht, wt, (sx, sy), shift), len(images), "cpu")

    def _fit_field(self, out: dict, cam_id: str, ht: int, wt: int, scale, shift) -> Tensor:
        theta = out["theta"].double()
        if self.net.field_repr == "logmap":
            theta = geometry.field_from_logmap(theta)
        x = np.ascontiguousarray(geometry.grid_knot_pixels(ht, wt, self.net.k, dtype=torch.float64).numpy())
        cxcy = None if self.variant == "free" else np.array((0.5 * wt, 0.5 * ht))
        p = fit_rays(cam_id, field_rays(theta.cpu().numpy()), x, self.iters, cxcy=cxcy,
                     simple=self.variant == "square", device=self.fit_device)
        p[:, 0] /= scale[0]
        p[:, 1] /= scale[1]
        p[:, 2] = (p[:, 2] - shift[0]) / scale[0]
        p[:, 3] = (p[:, 3] - shift[1]) / scale[1]
        return torch.from_numpy(p)

    def _fit_knots(self, out: dict, cam_id: str, ht: int, wt: int, scale, shift) -> Tensor:
        if REGRESSION_CAMS.get(self.profile) == cam_id:
            rho_max = geometry.rho_max_of(ht, wt)
            f = out["log_f"].double().cpu().exp() * rho_max
            k1 = geometry.division_k1(f, out["kappa"].double().cpu(), rho_max)
            params = fit.params_to_anycalib(cam_id, f, k1[:, None], ht, wt)
        else:
            theta = out["theta"].double().cpu()
            rho = geometry.knot_radii(ht, wt, self.net.k, self.net.placement, dtype=torch.float64)
            params = fit.fit_profile(cam_id, rho.expand(len(theta), -1), theta, None, h=ht, w=wt,
                                     gn_iters=self.iters)
        cam = CameraFactory.create_from_id(cam_id)
        return cam.reverse_scale_and_shift(params, torch.tensor(scale, dtype=torch.float64),
                                           torch.tensor(shift, dtype=torch.float64))

    def forward_network(self, image: Tensor):
        return self.net(image)


class AnyCalibNetwork:
    """AnyCalib's network, run once per batch however many predictors fit its field."""

    def __init__(self, spec: str, device: str):
        from litecalib.anycalib.model.anycalib_pretrained import AnyCalib

        if Path(spec).exists():
            config, state = load_checkpoint(spec)
            assert config["arch"] == "anycalib", f"{spec} is not an AnyCalib checkpoint"
            self.model = build_network(config, state)
            self.name = Path(spec).stem
        else:
            self.model = AnyCalib(model_id=spec)
            self.name = spec
        self.model.eval().to(device)
        self.device = device
        self._key, self._value = None, None

    def net_forward(self, im: Tensor) -> dict:
        m = self.model
        return m.head(m.decoder(m.backbone(im)))

    @torch.no_grad()
    def field(self, images: list[Tensor]) -> tuple[dict, Tensor, Tensor, Tensor]:
        """(the dense field, the resized image, scale_xy, shift_xy), cached for the batch `images`."""
        if self._key is images:
            return self._value
        m = self.model
        im = torch.stack(images).to(self.device)
        ho, wo = im.shape[-2:]
        target_ar = max(m.AR_RANGE[0], min(ho / wo, m.AR_RANGE[1]))
        im, scale_xy, shift_xy = m.set_im_size(im, m.compute_target_size(m.RESOLUTION, target_ar))
        out = self.net_forward(im)
        b, _, h, w = im.shape
        out["rays"] = out["rays"].permute(0, 2, 3, 1).reshape(b, h * w, 3)
        out["tangent_coords"] = out["tangent_coords"].permute(0, 2, 3, 1).reshape(b, h * w, 2)
        self._key, self._value = images, (out, im, scale_xy, shift_xy)
        return self._value


class AnyCalibPredictor(Timed):
    """An AnyCalib network with a calibrator: the released one on the dense field or an `egrid` sampling, or the
    compiled one (``acnb_*``). `c_mode`: ``free``, ``fixed`` (c at the image center, `@cfix`) or ``square``
    (c fixed and one focal, `@csquare`)."""

    def __init__(self, network: AnyCalibNetwork, sampling: str = sparse.DENSE, c_mode: str = "free",
                 fit_device: str | None = None):
        self.network = network
        kind, n, iters = sparse.parse_sampling(sampling)
        self.compiled = kind.startswith("acnb_")
        self.fit_device = fit_device or network.device
        if self.compiled:
            self.calibrator = sparse.CompiledCalibrator(n, iters, self.fit_device)
        elif kind == sparse.DENSE:
            self.calibrator = network.model.calibrator
        else:
            self.calibrator = sparse.PixelCalibrator(network.model.calibrator, n)
        assert c_mode in ("free", "fixed", "square"), c_mode
        self.c_mode = c_mode
        self.name = (network.name + ("" if kind == sparse.DENSE else f"@{sparse.sampling_name(sampling)}")
                     + {"free": "", "fixed": "@cfix", "square": "@csquare"}[c_mode]
                     + ("@cuda" if self.compiled and self.fit_device.startswith("cuda") else ""))
        self.info = {"type": "anycalib", "model": network.name, "sampling": sampling, "c": c_mode,
                     "fit_device": self.fit_device}
        self._init_timing()

    @torch.no_grad()
    def __call__(self, images: list[Tensor], cam_id: str, h: int, w: int) -> Tensor:
        out, im, scale_xy, shift_xy = self.network.field(images)
        data = {"image": im, "cam_id": [cam_id] * len(images)}
        if self.c_mode != "free":
            center = torch.tensor((w / 2, h / 2), dtype=torch.float64)
            center = center * scale_xy.double().cpu() + shift_xy.double().cpu()
            data["cxcy"] = (np.ascontiguousarray(np.broadcast_to(center.numpy(), (len(images), 2)))
                            if self.compiled else center.to(out["rays"]).expand(len(images), 2))
        torch_simple = self.c_mode == "square" and not self.compiled
        if torch_simple:
            data["cam_id"] = [f"simple_{cam_id}"] * len(images)
        elif self.c_mode == "square":
            data["simple"] = True
        if self.fit_device != self.network.device and not self.compiled:
            out = {k: out[k].to(self.fit_device) for k in ("rays", "tangent_coords")}
            data = {**data, "image": im.to(self.fit_device),
                    **({"cxcy": data["cxcy"].to(self.fit_device)} if "cxcy" in data else {})}
        intrinsics = self._timed(lambda: self.calibrator(out, data)["intrinsics"], len(images), self.fit_device)
        if torch_simple:
            intrinsics = [torch.cat((p[:1], p)) for p in intrinsics]
        cam = CameraFactory.create_from_id(cam_id)
        intrinsics = torch.stack([cam.reverse_scale_and_shift(p, scale_xy.to(p), shift_xy.to(p))
                                  for p in intrinsics])
        return intrinsics.double().cpu()

    def forward_network(self, image: Tensor):
        return self.network.net_forward(image)


def build_predictors(spec: str, device: str, ours_fits=(OURS_FIT_DEFAULT,), samplings=(sparse.DENSE,),
                     c_modes=("free",), fit_devices=(None,)) -> list:
    """The predictors of one ``--model``: a LiteCalib checkpoint (one per fit and fit device), or an AnyCalib model
    (released id or checkpoint; one per sampling, c mode and fit device, sharing one forward per batch)."""
    is_anycalib = spec.startswith("anycalib_") and not Path(spec).exists()
    if not is_anycalib:
        is_anycalib = load_checkpoint(spec)[0]["arch"] == "anycalib"
    if is_anycalib:
        network = AnyCalibNetwork(spec, device)
        out = {}
        for s in samplings:
            for c in c_modes:
                for fd in fit_devices:
                    p = AnyCalibPredictor(network, s, c, fd)
                    out.setdefault(p.name, p)
        return list(out.values())
    preds = []
    for f in ours_fits:
        for fd in fit_devices:
            fd = fd or "cpu"
            try:
                preds.append(OursPredictor(Path(spec), device, f, fd))
            except ValueError as e:
                print(f"skipping {spec} @ {f}: {e}", flush=True)
    if not preds:
        raise SystemExit(f"no --fit applies to {spec}")
    return preds


def score_batch(cam_id: str, preds: dict[str, Tensor], sub: dict, device: str) -> dict[str, list[dict]]:
    """AnyCalib's per-image metrics of every predictor's intrinsics for one batch."""
    cam = CameraFactory.create_from_id(cam_id)
    scale = torch.ones(2, dtype=torch.float64, device=device)
    shift = torch.zeros(2, dtype=torch.float64, device=device)
    preds = {name: pred.float().to(device) for name, pred in preds.items()}
    out: dict[str, list[dict]] = {name: [] for name in preds}
    for j, p_gt in enumerate(sub["params"].to(device)):
        gt = metrics_lib.GroundTruth(cam, sub["cam_id"], p_gt, sub["w"], sub["h"])
        for name, pred in preds.items():
            out[name].append(metrics_lib.metrics(pred[j].clone(), gt, cam, scale, shift,
                                                 metrics_lib.PIXEL_THRESHOLDS))
    return out


def evaluate_models(predictors: dict, groups: list[dict], fit_cam: str, batch_size: int, device: str) -> dict:
    """Score every predictor on `groups`, batch by batch: {name: summary}."""
    per_image: dict[str, list[dict]] = {name: [] for name in predictors}
    fit_start = {name: (p.fit_seconds, p.fit_images) for name, p in predictors.items()}
    total = sum(len(e["files"]) for e in groups)
    n, t_start, last = 0, time.time(), time.time()
    for entry in groups:
        cam = entry["cam_id"] if fit_cam == metrics_lib.FIT_GT else fit_cam
        for i in range(0, len(entry["files"]), batch_size):
            sl = slice(i, i + batch_size)
            images = [data_lib.read_image(dp) for dp in entry["files"][sl]]
            preds = {name: p(images, cam, entry["h"], entry["w"]) for name, p in predictors.items()}
            scored = score_batch(cam, preds, {**entry, "params": entry["params"][sl]}, device)
            for name in predictors:
                per_image[name] += scored[name]
            n += len(images)
            if time.time() - last >= 60 or n == total:
                last = time.time()
                print(f"  {n}/{total} images, {n / (last - t_start):.2f} img/s", flush=True)
    summaries = {}
    for name, p in predictors.items():
        summary = metrics_lib.summarize(per_image[name])
        fs, fi = p.fit_seconds - fit_start[name][0], p.fit_images - fit_start[name][1]
        summary["fit_ms_per_image"] = round(1000 * fs / max(1, fi), 3)
        summaries[name] = summary
    return summaries


@torch.no_grad()
def efficiency(predictor, h: int = 280, w: int = 364, runs: int = 30, device: str = "cuda") -> dict:
    """Parameters, MACs (torch's FlopCounterMode) and fp16 batch-1 latency of the network alone."""
    net = predictor.net if isinstance(predictor, OursPredictor) else predictor.network.model
    params = sum(p.numel() for n, p in net.named_parameters() if not n.startswith("dense."))
    out: dict = {"params_M": round(params / 1e6, 2)}
    hn, wn = target_size(h, w)
    image = torch.rand(1, 3, hn, wn, device=device)
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as counter:
            predictor.forward_network(image)
        out["MACs_G"] = round(counter.get_total_flops() / 2 / 1e9, 2)
    except Exception as e:
        out["MACs_G"], out["MACs_error"] = None, str(e)
    if device.startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.float16):
            for _ in range(5):
                predictor.forward_network(image)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(runs):
                predictor.forward_network(image)
            torch.cuda.synchronize()
        out["latency_ms"] = round(1000 * (time.time() - t0) / runs, 2)
    out["input"] = f"{hn}x{wn}"
    return out


def markdown(results: dict) -> str:
    """The results as a markdown table."""
    cols = [("median_ang_error", "median AE (deg)", "{:.3f}"), ("auc_ang_error@5", "AE mAA@5", "{:.3f}"),
            ("median_reproj_error", "median RE (px)", "{:.2f}"), ("median_vfov_error", "median vFoV (deg)", "{:.3f}"),
            ("median_hfov_error", "median hFoV (deg)", "{:.3f}"), ("fit_ms_per_image", "fit (ms)", "{:.2f}")]
    lines = ["| dataset | fit model | model | " + " | ".join(c[1] for c in cols) + " | n |",
             "|" + "---|" * (len(cols) + 4)]
    for dset, models in results["results"].items():
        for m, s in models.items():
            cells = " | ".join(f.format(s[k]) if k in s else "-" for k, _, f in cols)
            lines.append(f"| {dset} | `{results['datasets'][dset]['fit_cam']}` | {m} | {cells} | {s['n']} |")
    if results.get("efficiency"):
        lines += ["", "| model | params (M) | MACs (G) | fp16 latency (ms) | input |", "|---|---|---|---|---|"]
        lines += [f"| {m} | {e['params_M']} | {e.get('MACs_G')} | {e.get('latency_ms', '-')} | {e['input']} |"
                  for m, e in results["efficiency"].items()]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", required=True,
                    help="LiteCalib checkpoints (.pt or released .safetensors), AnyCalib checkpoints, or "
                         "anycalib_{pinhole,gen,dist,edit}")
    ap.add_argument("--data", nargs="+", default=["stanford2d3d"],
                    help="benchmark names, or `dataset:split` for a generated view dataset")
    ap.add_argument("--fit_cam", default=None,
                    help="camera model to fit (default: each benchmark's own, metrics.EVAL_CAM)")
    ap.add_argument("--fit", nargs="+", default=[OURS_FIT_DEFAULT],
                    help="LiteCalib checkpoints: ac{square,held,free}_it<N> (field) or it<N> (radial)")
    ap.add_argument("--anycalib_sampling", nargs="+", default=[sparse.DENSE],
                    help="AnyCalib models: dense, egrid[:N], acnb_dense:it<K>, acnb_egrid[:N]:it<K> (eval/sparse.py)")
    ap.add_argument("--anycalib_c", nargs="+", default=["free"], choices=["free", "fixed", "square"],
                    help="AnyCalib models: estimate c (free), hold it at the center (fixed), or hold it with one "
                         "focal (square)")
    ap.add_argument("--fit_device", nargs="+", default=[None],
                    help="where the fits run: cpu / cuda (default: cpu for LiteCalib, the network's device for "
                         "AnyCalib's calibrator)")
    ap.add_argument("--root", type=Path, default=None, help="benchmark root; default: paths config")
    ap.add_argument("--paths", default=None, help="machine in configs/paths (default: $LITECALIB_PATHS or local)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N images per dataset (smoke test)")
    ap.add_argument("--efficiency", action="store_true", help="also report params / MACs / latency")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch_threads", type=int, default=None,
                    help="torch's CPU threads (default: half the CPUs, leaving the rest to Numba's fits)")
    ap.add_argument("--tag", default="eval", help="run name; used in --out and the SLURM log path")
    ap.add_argument("--out", type=Path, default=None, help="default: results/eval_<tag>.md (+ .json)")
    slurm.add_arguments(ap, job_name="litecalib-eval", log_dir="logs/eval_submit/%t/%j", timeout_min=60)
    return ap


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = ROOT / "results" / f"eval_{args.tag}.md"
    if args.submitit:
        slurm.submit(args, argv, ap, module="evaluate", module_dir=Path(__file__).resolve().parent)
        return
    torch.set_grad_enabled(False)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    torch.set_num_threads(args.torch_threads or max(1, (cpus or 2) // 2))
    device = slurm.device_or_cpu(args.device)
    predictors = {}
    for spec in args.model:
        for p in build_predictors(spec, device, tuple(args.fit), tuple(args.anycalib_sampling),
                                  tuple(args.anycalib_c), tuple(args.fit_device)):
            predictors[p.name] = p
            print(f"model {p.name}: {p.info}", flush=True)

    results: dict = {"models": {n: p.info for n, p in predictors.items()}, "datasets": {},
                     "results": defaultdict(dict), "device": device}
    for spec in args.data:
        label, datapoints = load_datapoints(spec, args.root, args.paths)
        groups = group_by_shape(datapoints, args.limit)
        name = Path(parse_spec(label)[0]).name
        fit_cam = args.fit_cam or metrics_lib.EVAL_CAM.get(name)
        if fit_cam is None:
            raise SystemExit(f"no default camera model for {name!r}; pass --fit_cam")
        results["datasets"][label] = {"fit_cam": fit_cam, "images": sum(len(g["files"]) for g in groups),
                                      "cam_ids_gt": sorted({g["cam_id"] for g in groups})}
        print(f"\n{label}: {results['datasets'][label]}", flush=True)
        for model_name, s in evaluate_models(predictors, groups, fit_cam, args.batch_size, device).items():
            results["results"][label][model_name] = s
            print(f"  {model_name:<40} median AE {s['median_ang_error']:7.3f} deg  "
                  f"mAA@5 {100 * s['auc_ang_error@5']:5.1f}  vFoV {s['median_vfov_error']:7.3f} deg  "
                  f"fit {s['fit_ms_per_image']:.2f} ms", flush=True)
    if args.efficiency:
        results["efficiency"] = {n: efficiency(p, device=device) for n, p in predictors.items()}
    results["results"] = dict(results["results"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown(results), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()

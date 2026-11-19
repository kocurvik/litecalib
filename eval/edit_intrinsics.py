"""AnyCalib's Table 4 metrics on the `_edit` benchmarks: the focal and principal point errors of a free pinhole
fit.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import slurm

SETS = ["stanford2d3d_edit", "tartanair_edit", "lamar2k_edit"]
FIT = "acfree_it2"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "edit_intrinsics.json"))
    slurm.add_arguments(ap, job_name="edit-intrinsics", log_dir="logs/eval_submit/%t/%j", timeout_min=120)
    args = ap.parse_args(argv)
    if args.submitit:
        slurm.submit(args, argv, ap, module="edit_intrinsics", module_dir=Path(__file__).resolve().parent, tag="edit_intrinsics")
        return

    import numpy as np
    import torch

    import data as data_lib
    import evaluate

    torch.set_grad_enabled(False)
    device = slurm.device_or_cpu("cuda")
    out = {}
    for spec in args.model:
        predictor = evaluate.build_predictors(spec, device, ours_fits=(FIT,))[0]
        name = Path(spec).stem
        out[name] = {}
        for ds in SETS:
            _, points = evaluate.load_datapoints(ds, None, None)
            f_errors, c_errors, f_signed = [], [], []
            for group in evaluate.group_by_shape(points, args.limit):
                assert group["cam_id"].startswith("pinhole"), group["cam_id"]
                h, w = group["h"], group["w"]
                for i in range(0, len(group["files"]), args.batch_size):
                    images = [data_lib.read_image(dp) for dp in group["files"][i:i + args.batch_size]]
                    pred = predictor(images, "pinhole", h, w).numpy()
                    gt = group["params"][i:i + args.batch_size].numpy()
                    f_errors.append(np.abs(gt[:, :2] - pred[:, :2]) / gt[:, :2])
                    c_errors.append(2 * np.abs(gt[:, 2:4] - pred[:, 2:4]) / np.array([w, h]))
            f_all, c_all = np.concatenate(f_errors), np.concatenate(c_errors)
            out[name][ds] = {
                "n": len(f_all),
                "ef_pooled": float(np.median(f_all)), "ec_pooled": float(np.median(c_all)),
                "ef_image_mean": float(np.median(f_all.mean(1))), "ec_image_mean": float(np.median(c_all.mean(1))),
                "ef_image_max": float(np.median(f_all.max(1))), "ec_image_max": float(np.median(c_all.max(1))),
                "ec_norm": float(np.median(np.linalg.norm(c_all, axis=1))),
            }
            print(name, ds, out[name][ds], flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

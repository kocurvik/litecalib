"""Sanity-check a generated view dataset before training on it."""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from math import pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vendor import ANYCALIB_ROOT

import data
from litecalib import geometry
import gt
from litecalib.anycalib.cameras import CameraFactory

RAD2DEG = 180 / pi
Image.MAX_IMAGE_PIXELS = None

RECIPE_OF = {
    "openpano_v2": "openpano_v2.yaml",
    "openpano_v2_radial": "openpano_v2_radial.yaml",
    "openpano_v2_dist": "openpano_v2_dist.yaml",
    "openpano_v2_gen": "openpano_v2_gen.yaml",
    "commons_p": "openpano_v2.yaml",
    "commons_gen": "openpano_v2_gen.yaml",
    "commons_sa_gen": "openpano_v2_gen.yaml",
}


def git_hash(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def recipe_ranges(dataset_dir: Path) -> dict:
    """The distributions the generation config declares, as {name: (low, high)}."""
    name = RECIPE_OF.get(dataset_dir.name)
    if name is None:
        return {}
    import yaml

    conf = yaml.safe_load((ANYCALIB_ROOT / "siclib/datasets/configs" / name).read_text())
    out: dict[str, tuple[float, float]] = {}

    def add(key: str, spec: dict) -> None:
        if not isinstance(spec, dict):
            return
        opts = spec.get("options", {})
        if spec.get("type") == "uniform":
            out[key] = (opts["loc"], opts["loc"] + opts["scale"])
        elif spec.get("type") == "truncnorm":
            out[key] = (opts["a"] * opts["scale"] + opts["loc"], opts["b"] * opts["scale"] + opts["loc"])

    for key, spec in (conf.get("parameter_dists") or {}).items():
        add(key, spec)
    for key in ("roll", "pitch", "resize_factor"):
        if key in conf:
            add(key, conf[key])
    for spec in conf.get("intrinsics") or []:
        add(f"{spec['cam_id']}/vfov", spec.get("vfov", {}))
        for dist in spec.get("dist") or []:
            add(f"{spec['cam_id']}/{dist['name']}", dist)
    return out


def index_stats(dataset_dir: Path, split: str, k: int, limit: int | None) -> dict:
    """Counts, intrinsics distributions and per-view validity, from the index alone."""
    points = data.load_index(dataset_dir, split)[:limit]
    if not points:
        return None
    per_cam: dict[str, list] = defaultdict(list)
    for dp in points:
        per_cam[str(dp.cam_id)].append(dp)

    panos = Counter(dp.name.rsplit("_", 1)[0] for dp in points)
    sizes = Counter((dp.h, dp.w) for dp in points)
    out: dict = {
        "split": split,
        "views": len(points),
        "panoramas": len(panos),
        "views_per_panorama": {
            "min": min(panos.values()), "max": max(panos.values()),
            "mean": round(sum(panos.values()) / len(panos), 2),
        },
        "image_sizes": {f"{h}x{w}": n for (h, w), n in sizes.most_common()},
        "cam_ids": {c: len(v) for c, v in sorted(per_cam.items())},
        "per_cam": {},
        "samples": {},
    }

    for cam_id, dps in sorted(per_cam.items()):
        cam = CameraFactory.create_from_id(cam_id)
        params = torch.from_numpy(np.stack([dp.params for dp in dps])).double()
        h, w = dps[0].h, dps[0].w
        same_size = all((dp.h, dp.w) == (h, w) for dp in dps)

        vfov, valid_vfov = cam.get_vfov(params, h)
        centered = gt.check_center_and_aspect(cam_id, params, h, w) if same_size else None
        _, _, profile_ok = gt.gt_knots(cam_id, params, h, w, k, check_invariants=False)
        _, _, profile_ok_dense = gt.gt_knots(cam_id, params, h, w, 4 * k, check_invariants=False)

        stats = {
            "n": len(dps),
            "vfov_deg": summary(RAD2DEG * vfov[valid_vfov] if valid_vfov.any() else vfov),
            "invalid_vfov": int((~valid_vfov).sum()),
            "centered_and_square": None if centered is None else int(centered.sum()),
            "profile_valid": int(profile_ok.sum()),
            "profile_valid_dense": int(profile_ok_dense.sum()),
            "dropped_in_training": int((~profile_ok).sum()),
        }
        dist = cam.params_to_dict(params)["k"]
        for i, key in enumerate(distortion_names(cam_id)):
            stats[key] = summary(dist[:, i])
        out["per_cam"][cam_id] = stats
        out["samples"][cam_id] = [dp.name for dp in dps[:3]]
    return out


def distortion_names(cam_id: str) -> list[str]:
    name, num_k = geometry.parse_cam_id(cam_id)
    return {
        "pinhole": [],
        "radial": [f"k{i + 1}" for i in range(num_k)],
        "kb": [f"k{i + 1}" for i in range(num_k)],
        "division": [f"k{i + 1}" for i in range(num_k)],
        "ucm": ["xi"],
        "eucm": ["alpha", "beta"],
        "fov": ["omega"],
    }[name]


def summary(x: torch.Tensor) -> dict:
    x = x.double().flatten()
    q = torch.quantile(x, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], dtype=torch.float64))
    return {
        "min": round(float(q[0]), 5), "p1": round(float(q[1]), 5),
        "median": round(float(q[2]), 5), "p99": round(float(q[3]), 5),
        "max": round(float(q[4]), 5), "mean": round(float(x.mean()), 5),
    }


def check_loader(dataset_dir: Path, split: str, k: int, batches: int, batch_size: int) -> dict:
    """Run real training batches: the centered resize, the AR sampling, the knot GT."""
    ds = data.build_dataset(dataset_dir, split, k=k, check_invariants=True)
    loader = data.make_loader(ds, batch_size, "train", num_workers=0)
    shapes, invalid, seen = [], 0, 0
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        h, w = batch["image"].shape[-2:]
        shapes.append(f"{int(h)}x{int(w)}")
        invalid += int((~batch["valid"]).sum())
        seen += len(batch["valid"])
        assert batch["theta"].shape == (len(batch["valid"]), k)
        assert (batch["theta"].diff(dim=-1)[batch["valid"]] > 0).all(), "profile not increasing"
        assert int(h) % 14 == 0 and int(w) % 14 == 0
    return {"batches": len(shapes), "views": seen, "invalid": invalid, "shapes": shapes}


def read_resize_factors(dataset_dir: Path, split: str) -> dict[str, float]:
    """The per-view `resize_factor` the generator sampled, from the index itself."""
    csv = dataset_dir / f"{split}.csv"
    if csv.exists():
        import pandas as pd

        df = pd.read_csv(csv)
        if "resize_factor" in df.columns:
            return dict(zip(df["fname"], df["resize_factor"].astype(float)))
        return {}
    h5 = dataset_dir / f"{split}.h5"
    if h5.exists():
        import h5py

        with h5py.File(h5, "r") as f:
            if f.attrs.get("format") == data.STORE_FORMAT:
                if "resize_factor" not in f:
                    return {}
                return dict(zip(f["fname"].asstr()[...], f["resize_factor"][...].astype(float)))
            return {
                k: float(g.attrs["resize_factor"]) for k, g in f.items()
                if "resize_factor" in g.attrs
            }
    return {}


def check_panorama_resolution(
    dataset_dir: Path, split: str, pano_dir: Path, limit: int | None, resolution: int = 102_400
) -> dict:
    """How many views were rendered by upsampling their panorama (see the module docstring)."""
    points = data.load_index(dataset_dir, split)[:limit]
    res_fac = read_resize_factors(dataset_dir, split)
    heights: dict[str, int] = {}
    scales, vfovs, missing = [], [], 0
    input_scales: list[float] = []
    index = {p.stem: p for p in Path(pano_dir).glob("*.*")}
    for dp in points:
        stem = dp.name.rsplit("_", 1)[0]
        if stem not in heights:
            path = index.get(stem)
            if path is None:
                missing += 1
                heights[stem] = 0
                continue
            with Image.open(path) as im:
                heights[stem] = im.size[1]
        h_pano = heights[stem]
        if not h_pano:
            continue
        cam = CameraFactory.create_from_id(str(dp.cam_id))
        vfov, ok = cam.get_vfov(torch.from_numpy(dp.params).double()[None], dp.h)
        if not bool(ok.all()):
            continue
        scale = float(pi / vfov[0] * dp.h / h_pano) * res_fac.get(dp.name, 1.0)
        scales.append(scale)
        vfovs.append(RAD2DEG * float(vfov[0]))
        input_scales.append(scale * (resolution / (dp.h * dp.w)) ** 0.5)

    arr = np.array(scales) if scales else np.zeros(1)
    at_input = np.array(input_scales) if input_scales else np.zeros(1)
    vf = np.array(vfovs) if vfovs else np.zeros(1)
    out = {
        "views_checked": len(scales),
        "panoramas_missing": missing,
        "resize_factor_applied": bool(res_fac),
        "resize_factor_median": round(float(np.median(list(res_fac.values()))), 4) if res_fac else None,
        "scale_median": round(float(np.median(arr)), 4),
        "scale_p99": round(float(np.percentile(arr, 99)), 4),
        "upsampled_%": round(100 * float((arr > 1).mean()), 2),
        "upsampled_1.5x_%": round(100 * float((arr > 1.5).mean()), 2),
        "network_resolution": resolution,
        "scale_at_input_median": round(float(np.median(at_input)), 4),
        "upsampled_at_input_%": round(100 * float((at_input > 1).mean()), 2),
        "note": "scale > 1 means the panorama was upsampled, so the view is soft. The scale "
                "is ~1/vfov, so blur correlates with a long focal length - a cue a network "
                "can learn and then fail on sharp real images. `*_at_input` is the same "
                "measured after the isotropic downscale to `network_resolution` pixels, "
                "which cancels part of it and is what the network actually sees.",
    }
    if len(vf) > 4:
        edges = np.quantile(vf, [0, 0.25, 0.5, 0.75, 1.0])
        for key, values in (("", arr), ("_at_input", at_input)):
            out[f"upsampled{key}_%_by_vfov_quartile"] = {
                f"{edges[i]:.0f}-{edges[i + 1]:.0f} deg": round(
                    100 * float((values[(vf >= edges[i]) & (vf <= edges[i + 1])] > 1).mean()), 2
                )
                for i in range(4)
            }
    return out


def histograms(dataset_dir: Path, splits: list[str], limit: int | None, out: Path) -> None:
    """vFoV and distortion histograms per split, with the recipe's declared range marked."""
    ranges = recipe_ranges(dataset_dir)
    series: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for split in splits:
        per_cam: dict[str, list] = defaultdict(list)
        for dp in data.load_index(dataset_dir, split)[:limit]:
            per_cam[str(dp.cam_id)].append(dp)
        for cam_id, dps in per_cam.items():
            cam = CameraFactory.create_from_id(cam_id)
            params = torch.from_numpy(np.stack([dp.params for dp in dps])).double()
            vfov, ok = cam.get_vfov(params, dps[0].h)
            series[f"{cam_id} vfov (deg)"][split] = (RAD2DEG * vfov[ok]).numpy()
            dist = cam.params_to_dict(params)["k"]
            for i, key in enumerate(distortion_names(cam_id)):
                series[f"{cam_id} {key}"][split] = dist[:, i].numpy()

    if not series:
        return
    cols = min(3, len(series))
    rows = (len(series) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    for ax, (title, per_split) in zip(axs.flat, series.items()):
        for split, values in per_split.items():
            ax.hist(values, bins=60, alpha=0.55, label=f"{split} ({len(values)})", density=True)
        cam_id, _, key = title.partition(" ")
        key = key.split(" ")[0]
        lo_hi = ranges.get(f"{cam_id}/{key}") or ranges.get(key)
        if lo_hi and key == "vfov":
            lo_hi = (RAD2DEG * lo_hi[0], RAD2DEG * lo_hi[1])
        if lo_hi:
            for x in lo_hi:
                ax.axvline(x, color="k", ls="--", lw=1)
            ax.set_title(f"{title}\nrecipe: [{lo_hi[0]:.3g}, {lo_hi[1]:.3g}]", fontsize=9)
        else:
            ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    for ax in axs.flat[len(series) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def thumbnail_sheet(dataset_dir: Path, split: str, n: int, out: Path, thumb: int = 160) -> None:
    """A sheet of views per camera model, for the eyeball check asks for."""
    per_cam: dict[str, list] = defaultdict(list)
    for dp in data.load_index(dataset_dir, split):
        if len(per_cam[str(dp.cam_id)]) < n:
            per_cam[str(dp.cam_id)].append(dp)
    if not per_cam:
        return
    rows = sorted(per_cam)
    fig, axs = plt.subplots(len(rows), n, figsize=(1.6 * n, 1.8 * len(rows)), squeeze=False)
    for r, cam_id in enumerate(rows):
        for c in range(n):
            ax = axs[r][c]
            ax.axis("off")
            if c >= len(per_cam[cam_id]):
                continue
            dp = per_cam[cam_id][c]
            im = Image.fromarray(data.read_image_uint8(dp))
            im.thumbnail((thumb, thumb))
            ax.imshow(np.asarray(im))
            cam = CameraFactory.create_from_id(cam_id)
            vfov, ok = cam.get_vfov(torch.from_numpy(dp.params).double()[None], dp.h)
            ax.set_title(
                f"{cam_id}\n{RAD2DEG * float(vfov[0]):.0f}°" if bool(ok.all()) else cam_id,
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset directory, or a name under the panorama roots")
    ap.add_argument("--paths", default=None, help="machine in configs/paths (default: auto)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--K", type=int, default=geometry.DEFAULT_K)
    ap.add_argument("--limit", type=int, default=None, help="only the first N views per split")
    ap.add_argument("--panoramas", type=Path, default=None,
                    help="the split's panorama directory, to check for upsampled views")
    ap.add_argument("--resolution", type=int, default=102_400,
                    help="training resolution, for the upsampling check at the network's input")
    ap.add_argument("--thumbs", type=int, default=8, help="views per camera model in the sheet (0 to skip)")
    ap.add_argument("--loader_batches", type=int, default=4, help="real training batches to run (0 to skip)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dataset_dir = data.resolve_dataset_dir(args.data, args.paths)
    name = dataset_dir.name
    print(f"checking {dataset_dir}")

    report: dict = {
        "dataset": name,
        "path": str(dataset_dir),
        "recipe": RECIPE_OF.get(name, "unknown"),
        "K": args.K,
        "git": {"litecalib": git_hash(ROOT), "anycalib": git_hash(ANYCALIB_ROOT)},
        "recipe_ranges": recipe_ranges(dataset_dir),
        "splits": {},
    }

    present = [s for s in args.splits if (dataset_dir / f"{s}.csv").exists() or (dataset_dir / f"{s}.h5").exists()]
    missing = sorted(set(args.splits) - set(present))
    if missing:
        print(f"  no index for split(s) {missing} - skipping (Commons has no test split by design)")
    report["splits_missing"] = missing

    for split in present:
        stats = index_stats(dataset_dir, split, args.K, args.limit)
        if stats is None:
            print(f"\n{split}: empty index - skipping (Commons has no test split by design)")
            missing.append(split)
            continue
        report["splits"][split] = stats
        print(
            f"\n{split}: {stats['views']} views from {stats['panoramas']} panoramas "
            f"({stats['views_per_panorama']['mean']} per panorama, "
            f"min {stats['views_per_panorama']['min']}), sizes {stats['image_sizes']}"
        )
        for cam_id, s in stats["per_cam"].items():
            v = s["vfov_deg"]
            print(
                f"  {cam_id:<10} n={s['n']:<6} vFoV {v['min']:.1f}-{v['max']:.1f} deg "
                f"(median {v['median']:.1f})  centered+square {s['centered_and_square']}/{s['n']}"
                f"  profile valid {s['profile_valid']}/{s['n']}"
                + "".join(
                    f"  {key} [{s[key]['min']:.4g}, {s[key]['max']:.4g}]"
                    for key in distortion_names(cam_id)
                )
            )
            if s["dropped_in_training"]:
                print(f"    {s['dropped_in_training']} view(s) would be DROPPED in training")
            if s["centered_and_square"] not in (None, s["n"]):
                print(f"    WARNING: {s['n'] - s['centered_and_square']} view(s) are not centered/square")

    if args.loader_batches and present:
        split = "train" if "train" in present else present[0]
        report["loader"] = check_loader(dataset_dir, split, args.K, args.loader_batches, args.batch_size)
        print(f"\nloader ({split}): {report['loader']}")

    if args.panoramas and present:
        split = "train" if "train" in present else present[0]
        report["panorama_resolution"] = check_panorama_resolution(
            dataset_dir, split, args.panoramas, args.limit, args.resolution
        )
        print(f"\npanorama resolution ({split}): {report['panorama_resolution']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if present:
        histograms(dataset_dir, present, args.limit, args.out_dir / f"check_views_{name}.png")
        if args.thumbs:
            thumbnail_sheet(
                dataset_dir, "train" if "train" in present else present[0],
                args.thumbs, args.out_dir / f"check_views_{name}_thumbs.png",
            )
    out = args.out_dir / f"check_views_{name}.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""An `rd` radial-distortion collection -> an evaluation benchmark."""

from __future__ import annotations

import argparse
import math
import sys
from math import gcd
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import influx
import paths
import slurm

CAM_ID = "division:1"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
META_COLUMNS = (
    "fname", "source", "camera", "h", "w", "fx", "fy", "cx", "cy", "k1", "lambda_rd",
    "downscale", "h_src", "w_src", "theta_max_deg", "vfov_deg", "c_offset_px",
)


def read_calibration(src: Path) -> list[dict]:
    """Every calibrated image of an `rd` collection, as AnyCalib `division:1` cameras."""
    import h5py

    with h5py.File(src / "K.h5", "r") as fk, h5py.File(src / "parameters_rd.h5", "r") as fp:
        k_keys, p_keys = set(fk.keys()), {k[: -len("-hwK")] for k in fp.keys()}
        assert k_keys == p_keys, (
            f"{src}: K.h5 and parameters_rd.h5 disagree on {len(k_keys ^ p_keys)} names, "
            f"e.g. {sorted(k_keys ^ p_keys)[:3]}"
        )
        records = []
        for name in sorted(k_keys):
            K, P = fk[name][...], fp[f"{name}-hwK"][...]
            w, h, lam = float(P[0, 0]), float(P[1, 1]), float(P[2, 2])
            f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
            assert K[0, 0] == K[1, 1], f"{name}: fx != fy ({K[0, 0]} != {K[1, 1]})"
            records.append({
                "name": name,
                "h": int(h), "w": int(w),
                "params": [f, f, cx, cy, lam * (f / max(w, h)) ** 2],
                "lambda_rd": lam,
            })
    return records


def index_images(img_root: Path) -> dict[str, Path]:
    """Map each calibration name to its file: the names carry no extension, and a collection mixes `.jpg` and
    `.png` (`cathedral`: 1948 and 786).
    """
    index: dict[str, Path] = {}
    for path in img_root.rglob("*"):
        if path.suffix not in IMAGE_SUFFIXES or not path.is_file():
            continue
        key = path.relative_to(img_root).with_suffix("").as_posix().replace("/", "\\")
        assert key not in index, f"{img_root}: two files for {key!r} ({index[key]}, {path})"
        index[key] = path
    return index


def downscale_factor(h: int, w: int, target_side: int) -> int:
    """The integer factor bringing the long side closest to ``target_side``."""
    long_side = max(h, w)
    if long_side <= target_side:
        return 1
    g = gcd(h, w)
    divisors = [d for d in range(1, g + 1) if g % d == 0]
    return min(divisors, key=lambda d: (abs(long_side / d - target_side), d))


def camera_of(record: dict) -> tuple:
    """The camera a record belongs to: the calibration is per camera, not per image, so identical parameters
    *are* the grouping (`cathedral`: 2734 images, 42 cameras).
    """
    return (record["h"], record["w"], *(round(v, 9) for v in record["params"]))


def select(records: list[dict], per_camera: int) -> list[dict]:
    """At most ``per_camera`` evenly spaced frames from each camera (0 keeps everything)."""
    if per_camera <= 0:
        return records
    groups: dict[tuple, list[int]] = {}
    for i, record in enumerate(records):
        groups.setdefault(camera_of(record), []).append(i)
    keep = {i for indices in groups.values() for i in influx.even_sample(indices, per_camera)}
    return [record for i, record in enumerate(records) if i in keep]


def source_of(name: str) -> str:
    """The capture a name came from -- its directory, or ``<root>`` for the loose files."""
    return name.rsplit("\\", 1)[0] if "\\" in name else "<root>"


def encode(record: dict, path: Path, target_side: int, quality: int) -> tuple[dict, bytes]:
    """One image: decode, resize by an exact factor, re-encode, scale the intrinsics."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise IOError(f"could not decode {path}")
    h, w = record["h"], record["w"]
    assert image.shape[:2] == (h, w), (
        f"{record['name']}: {path} is {image.shape[1]}x{image.shape[0]}, the calibration "
        f"says {w}x{h}"
    )
    d = downscale_factor(h, w, target_side)
    ht, wt = h // d, w // d
    if d > 1:
        image = cv2.resize(image, (wt, ht), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, f"{record['name']}: JPEG encoding failed"

    params = influx.scale_params(CAM_ID, np.asarray(record["params"], dtype=np.float64), 1 / d)
    entry = {
        "fname": record["name"].replace("\\", "/") + ".jpg",
        "cam_id": CAM_ID,
        "h": ht, "w": wt,
        "params": [float(v) for v in params],
        "vfov": float(2 * math.atan(0.5 * ht / params[1])),
        "resize_factor": 1 / d,
        "source": source_of(record["name"]),
        "camera": "",
        "fx": float(params[0]), "fy": float(params[1]),
        "cx": float(params[2]), "cy": float(params[3]), "k1": float(params[4]),
        "lambda_rd": record["lambda_rd"],
        "downscale": d,
        "h_src": h, "w_src": w,
        "c_offset_px": float(np.hypot(params[2] - 0.5 * wt, params[3] - 0.5 * ht)),
    }
    return entry, jpeg.tobytes()


def prepare(args) -> None:
    import torch
    from concurrent.futures import ThreadPoolExecutor

    import data as data_lib
    import gt as gtmod

    src = Path(args.root) / args.dataset
    name = args.name or args.dataset
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_calibration(src)
    total, cameras = len(records), {camera_of(r) for r in records}
    records = select(records, args.per_camera)
    if args.limit:
        records = records[: args.limit]
    files = index_images(src / "images")
    missing = [r["name"] for r in records if r["name"] not in files]
    assert not missing, f"{src}: {len(missing)} calibrated images have no file, e.g. {missing[:3]}"

    order = {cam: i for i, cam in enumerate(sorted({camera_of(r) for r in records}))}
    print(f"{name}: {len(records)}/{total} images from {len(order)}/{len(cameras)} cameras "
          f"(cap {args.per_camera or 'none'} per camera), {src}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda r: encode(r, files[r["name"]], args.target_side, args.jpeg_quality), records
        ))
    for record, (entry, _) in zip(records, results):
        entry["camera"] = f"cam{order[camera_of(record)]:02d}"

    entries, blobs = [], []
    by_shape: dict[tuple[int, int], list[int]] = {}
    for i, (entry, _) in enumerate(results):
        by_shape.setdefault((entry["h"], entry["w"]), []).append(i)
    valid = np.ones(len(results), dtype=bool)
    theta_max = np.zeros(len(results))
    for (h, w), indices in by_shape.items():
        stacked = torch.tensor([results[i][0]["params"] for i in indices], dtype=torch.float64)
        _, tmax, ok = gtmod.gt_knots(CAM_ID, stacked, h, w, check_invariants=False)
        valid[indices], theta_max[indices] = ok.numpy(), np.degrees(tmax.numpy())
    for i, (entry, blob) in enumerate(results):
        if not valid[i]:
            print(f"  dropping {entry['fname']}: invalid profile", flush=True)
            continue
        entry["theta_max_deg"] = float(theta_max[i])
        entry["vfov_deg"] = math.degrees(entry["vfov"])
        entry["length"] = len(blob)
        entries.append(entry)
        blobs.append(blob)

    index = out_dir / "images.h5"
    def read(i: int) -> np.ndarray:
        return np.frombuffer(blobs[i], dtype=np.uint8)

    data_lib.write_store(index, entries, read, attrs={
        "name": name, "source": str(src), "dataset": args.dataset, "cam_id": CAM_ID,
        "per_camera": args.per_camera, "target_side": args.target_side,
        "jpeg_quality": args.jpeg_quality, "n_cameras": len(order),
        "note": "GT `c` is K.h5's, the distortion center of the division model is the image center"
    })
    influx.write_meta_csv(out_dir / "images_meta.csv", entries, META_COLUMNS)
    sizes = sorted({f"{e['w']}x{e['h']}" for e in entries})
    print(f"{name}: {len(entries)} images, {len(order)} cameras, sizes {sizes} -> {index}\n"
          + influx.summarise(entries, ("theta_max_deg", "c_offset_px", "k1", "downscale")),
          flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cathedral", help="collection under --root (default: cathedral)")
    p.add_argument("--per_camera", type=int, default=50,
                   help="evenly spaced frames kept per camera; 0 keeps all (default: 50)")
    p.add_argument("--target_side", type=int, default=1000,
                   help="long side the stored images aim at, reached only by exact integer "
                        "factors (default: 1000)")
    p.add_argument("--name", default=None, help="benchmark directory name (default: --dataset)")
    p.add_argument("--root", type=Path, default=None, help="rd collections root (default: paths config)")
    p.add_argument("--out", type=Path, default=None, help="benchmark root (default: paths config)")
    p.add_argument("--paths", default=None, help="machine in configs/paths (default: auto)")
    p.add_argument("--workers", type=int, default=8, help="threads decoding and re-encoding")
    p.add_argument("--jpeg_quality", type=int, default=influx.JPEG_QUALITY)
    p.add_argument("--limit", type=int, default=None, help="first N selected images (smoke test)")
    slurm.add_arguments(p, job_name="prepare-rd", log_dir="logs/prepare_rd/%t/%j", gpus=0,
                        partition="short", cpus_per_task=16, mem_gb=64, timeout_min=120)
    return p


def run(args) -> None:
    args.root = Path(args.root) if args.root else paths.get("rd_root", args.paths)
    args.out = Path(args.out) if args.out else paths.get("benchmarks", args.paths)
    prepare(args)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.submitit:
        slurm.submit(args, argv, parser, module="prepare_rd",
                     module_dir=Path(__file__).resolve().parent, tag=args.dataset)
        return
    run(args)


if __name__ == "__main__":
    main()

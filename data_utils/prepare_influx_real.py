"""InFlux-Real's validation split -> an evaluation benchmark."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import influx
import paths
import slurm

PARTITIONS = {
    "influx": ("video_frame_count_and_split_v1.json", "gt_validation_dict_v1.json", "influx_val"),
    "influx_pp_real": ("video_frame_count_and_split_v2.json", "gt_validation_dict_v2.json", "influx_pp_val"),
}
CAM_ID = "radial:2"
FIELDS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")
META_COLUMNS = (
    "fname", "partition", "video", "frame_index", "h", "w", "fx", "fy", "cx", "cy", "k1", "k2",
    "p1", "p2", "theta_max_deg", "vfov_deg", "c_offset_px", "tangential_px", "focal_length_mm",
    "focus_distance_m",
)


def probe(video: Path) -> tuple[int, int, int | None]:
    """``(w, h, nb_frames)`` from the container, without decoding it."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    nb = stream.get("nb_frames")
    nb = int(nb) if nb not in (None, "N/A") else None
    return int(stream["width"]), int(stream["height"]), nb


def extract_frames(video: Path, indices: list[int], size: tuple[int, int], dst: Path) -> list[Path]:
    """Decode exactly ``indices`` (0-based frame numbers) out of ``video``, scaled to ``size``."""
    dst.mkdir(parents=True, exist_ok=True)
    expr = "+".join(f"eq(n\\,{i})" for i in indices)
    w, h = size
    filter_graph = f"select={expr},scale={w}:{h}:flags=area"

    try:
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
                "-vf", filter_graph,
                "-fps_mode", "passthrough", "-an", "-sn", "-f", "image2",
                str(dst / "frame_%05d.png"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("=== FFmpeg Stderr Output ===")
        print(e.stderr)
        print("===========================")
        raise e

    return sorted(dst.glob("frame_*.png"))


def frame_sample(gt_video: dict, frame_count: int, n: int, field: str) -> list[int]:
    """The frame indices to keep from one video: ``n`` evenly spaced, usable ones."""
    usable = []
    for key, record in gt_video.items():
        values = record[field]
        if all(np.isfinite(values[f]) for f in FIELDS):
            usable.append(int(key))
    usable = sorted(i for i in usable if 0 <= i < frame_count)
    return influx.even_sample(usable, n)


def prepare(args, partition: str) -> None:
    import cv2

    import gt as gtmod
    import torch

    manifest_name, gt_name, default_name = PARTITIONS[partition]
    name = args.name or default_name
    out_dir = Path(args.out) / name
    cache = Path(args.cache) / partition
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(influx.download(
        influx.hf_url(influx.HF_REAL, f"{partition}/{manifest_name}"), cache / manifest_name
    ).read_text(encoding="utf-8"))
    ground_truth = json.loads(influx.download(
        influx.hf_url(influx.HF_REAL, f"{partition}/{gt_name}"), cache / gt_name
    ).read_text(encoding="utf-8"))

    videos = sorted(v for v, m in manifest.items() if m["split"] == "val")
    missing = [v for v in videos if v not in ground_truth]
    assert not missing, f"{partition}: {len(missing)} val videos without ground truth, e.g. {missing[:3]}"
    if args.limit:
        videos = videos[: args.limit]

    index = out_dir / "images.h5"
    log = influx.ShardLog(out_dir / "images.partial.jpeg", out_dir / "images.partial.jsonl", args.resume)
    if args.resume and index.exists() and not log.done:
        print(f"{name}: {index} exists and no journal - already complete", flush=True)
        log.close()
        return

    todo = [v for v in videos if v not in log.done]
    print(f"{name}: {len(videos)} val videos, {len(log.done)} already done, {len(todo)} to fetch",
          flush=True)
    start = time.time()
    for count, video in enumerate(todo, 1):
        frame_count = int(manifest[video]["frame_count"])
        wanted = frame_sample(ground_truth[video], frame_count, args.frames_per_video, args.intrinsics)
        if not wanted:
            print(f"  {video}: no usable ground truth - skipped", flush=True)
            log.add(video, [])
            continue
        requested = sorted(set(wanted) | {frame_count - 1})
        mp4 = cache / "videos" / f"{video}.mp4"
        influx.download(influx.hf_url(influx.HF_REAL, f"{partition}/videos/{video}.mp4"), mp4)
        w, h, nb = probe(mp4)
        assert nb is None or nb == frame_count, f"{video}: {nb} frames in the container, {frame_count} in the manifest"
        assert w % args.downscale == 0 and h % args.downscale == 0, \
            f"{video}: {w}x{h} is not divisible by --downscale {args.downscale}"
        wt, ht = w // args.downscale, h // args.downscale
        tmp = Path(tempfile.mkdtemp(prefix=f"influx_{video}_", dir=args.cache))
        try:
            files = extract_frames(mp4, requested, (wt, ht), tmp)
            assert len(files) == len(requested), (
                f"{video}: ffmpeg returned {len(files)} frames for {len(requested)} requested - "
                "the decoded frame numbering does not match the ground truth's"
            )
            by_index = dict(zip(requested, files))
            items, rows = [], []
            for i in wanted:
                image = cv2.imread(str(by_index[i]), cv2.IMREAD_COLOR)
                assert image is not None and image.shape[:2] == (ht, wt), f"{video}#{i}: {image is not None}"
                ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                assert ok, f"{video}#{i}: JPEG encoding failed"
                record = ground_truth[video][str(i)]
                full = np.array([record[args.intrinsics][f] for f in FIELDS], dtype=np.float64)
                params = influx.scale_params(CAM_ID, full[:6], 1 / args.downscale)
                lens = record.get("lens_metadata", {})
                entry = {
                    "fname": f"{video}/{i:06d}.jpg",
                    "cam_id": CAM_ID,
                    "h": ht, "w": wt,
                    "params": [float(v) for v in params],
                    "vfov": float(2 * math.atan(0.5 * ht / params[1])),
                    "resize_factor": 1 / args.downscale,
                    "partition": partition,
                    "video": video,
                    "frame_index": i,
                    **{f: float(v) for f, v in zip(FIELDS[:6], params)},
                    "p1": float(full[6]), "p2": float(full[7]),
                    "c_offset_px": float(np.hypot(params[2] - 0.5 * wt, params[3] - 0.5 * ht)),
                    "tangential_px": influx.max_tangential_px(params, full[6], full[7], ht, wt),
                    "focal_length_mm": float(lens.get("focal_length_mm", float("nan"))),
                    "focus_distance_m": float(lens.get("focus_distance_m", float("nan"))),
                }
                items.append((entry, jpeg.tobytes()))
                rows.append(entry)
            stacked = torch.tensor([r["params"] for r in rows], dtype=torch.float64)
            _, theta_max, valid = gtmod.gt_knots(CAM_ID, stacked, ht, wt, check_invariants=False)
            if not bool(valid.all()):
                bad = [r["fname"] for r, v in zip(rows, valid.tolist()) if not v]
                print(f"  {video}: dropping {len(bad)} frames with an invalid profile", flush=True)
            keep = []
            for entry, item, ok_i, tm in zip(rows, items, valid.tolist(), theta_max.tolist()):
                if ok_i:
                    entry["theta_max_deg"] = math.degrees(tm)
                    entry["vfov_deg"] = math.degrees(entry["vfov"])
                    keep.append(item)
            log.add(video, keep)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            if not args.keep_videos:
                mp4.unlink(missing_ok=True)
        rate = count / (time.time() - start)
        print(f"  [{count}/{len(todo)}] {video}: {len(log.done[video])} frames at {wt}x{ht} "
              f"({rate * 60:.1f} videos/min, eta {(len(todo) - count) / rate / 60:.0f} min)", flush=True)

    entries = [e for v in videos for e in log.done.get(v, [])]
    n = log.write_store(index, videos, attrs={
        "name": name, "source": influx.HF_REAL, "partition": partition, "split": "val",
        "cam_id": CAM_ID, "intrinsics": args.intrinsics, "downscale": args.downscale,
        "frames_per_video": args.frames_per_video, "jpeg_quality": args.jpeg_quality,
        "note": "tangential distortion (p1, p2) dropped; see images_meta.csv/tangential_px",
    })
    influx.write_meta_csv(out_dir / "images_meta.csv", entries, META_COLUMNS)
    print(f"{name}: {n} frames from {len(videos)} videos -> {index}\n"
          + influx.summarise(entries, ("theta_max_deg", "c_offset_px", "tangential_px")), flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subsets", nargs="+", dest="subsets", default=["influx", "influx_pp_real"],
                   choices=sorted(PARTITIONS), help="InFlux-Real partition(s) to prepare; `partition` is SLURM's")
    p.add_argument("--frames_per_video", type=int, default=30,
                   help="evenly spaced frames kept per video (default: 30)")
    p.add_argument("--downscale", type=int, default=2,
                   help="integer factor the 3424x2202 frames are stored at (default: 2)")
    p.add_argument("--intrinsics", default="intrinsics_gt",
                   choices=["intrinsics_gt", "intrinsics_gt_extrapolated"],
                   help="which ground truth to use; the extrapolated one has no NaNs but is "
                        "extrapolated past the calibration LUT")
    p.add_argument("--name", default=None, help="benchmark directory name (default: per partition)")
    p.add_argument("--out", type=Path, default=None, help="benchmark root (default: paths config)")
    p.add_argument("--cache", type=Path, default=None,
                   help="where videos are downloaded to and deleted from (default: the job's "
                        "`/work/<job id>`, else <out>/.influx_cache)")
    p.add_argument("--paths", default=None, help="machine in configs/paths (default: auto)")
    p.add_argument("--keep_videos", action="store_true", help="do not delete each .mp4 after extraction")
    p.add_argument("--jpeg_quality", type=int, default=influx.JPEG_QUALITY)
    p.add_argument("--limit", type=int, default=None, help="first N val videos (smoke test)")
    p.add_argument("--no_resume", dest="resume", action="store_false", help="ignore an existing journal")
    slurm.add_arguments(p, job_name="influx-real", log_dir="logs/influx_real/%t/%j", gpus=0,
                        partition="short", cpus_per_task=16, mem_gb=64, timeout_min=720)
    return p


def run(args) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe must be on PATH (conda install -c conda-forge ffmpeg)")
    args.out = Path(args.out) if args.out else paths.get("benchmarks", args.paths)
    args.cache = influx.cache_dir(Path(args.out) / ".influx_cache", args.cache)
    for partition in args.subsets:
        prepare(args, partition)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.submitit:
        slurm.submit(args, argv, parser, module="prepare_influx_real",
                     module_dir=Path(__file__).resolve().parent, tag="-".join(args.subsets))
        return
    run(args)


if __name__ == "__main__":
    main()

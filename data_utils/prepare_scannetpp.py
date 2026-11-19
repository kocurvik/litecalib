"""ScanNet++ -> AnyCalib's `scannetpp2k` benchmark (Table 3, `kb:4` ground truth)."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import influx
import paths
import slurm
import vendor

NAME = "scannetpp2k"
URL = "https://scannetpp.mlsg.cit.tum.de/scannetpp/download/v2?version=v1&token={token}&file={file}"
GT_H5 = vendor.ANYCALIB_ROOT / "siclib" / "eval" / "scannetpp2k_images.h5"


def read_token(args) -> str:
    token = os.environ.get("SCANNETPP_TOKEN", "").strip()
    if token:
        return token
    path = Path(args.token_file) if args.token_file else paths.get("scannetpp_token", args.paths)
    return path.read_text(encoding="utf-8").strip()


def url(token: str, file: str) -> str:
    return URL.format(token=token, file=file)


def check_token(token: str) -> None:
    """Fail at once, with the server's own words, on a bad or expired token -- rather than after
    `influx.download` has retried 355 zips into the same 401.
    """
    import requests

    r = requests.get(url(token, "splits/nvs_sem_train.txt"), timeout=60)
    if r.status_code != 200:
        msg = r.text.strip().replace(token, "<token>")[:500]
        raise SystemExit(f"ScanNet++ refused the token (HTTP {r.status_code}): {msg}")


def benchmark_images(limit: int | None = None) -> dict[str, list[str]]:
    """The benchmark's images, grouped by scene: ``{scene: [image, ...]}``."""
    import h5py

    scenes: dict[str, list[str]] = {}
    with h5py.File(GT_H5, "r") as f:
        for key in f.keys():
            scene, image = key.split("_", 1)
            scenes.setdefault(scene, []).append(image)
    order = sorted(scenes)[:limit] if limit else sorted(scenes)
    return {s: scenes[s] for s in order}


def fetch_scene(scene: str, images: list[str], token: str, img_dir: Path, cache: Path) -> list[str]:
    """One scene: download its zip, extract the benchmark's images from it, delete it."""
    targets = {im: img_dir / f"{scene}_{im}" for im in images}
    if all(t.exists() for t in targets.values()):
        return []
    zpath = cache / f"{scene}_resized_images.zip"
    influx.download(url(token, f"data/{scene}/dslr/resized_images.zip"), zpath)
    try:
        with zipfile.ZipFile(zpath) as z:
            by_name = {Path(m).name: m for m in z.namelist() if not m.endswith("/")}
            missing = [im for im in images if im not in by_name]
            for im, target in targets.items():
                if im in missing or target.exists():
                    continue
                part = target.with_name(target.name + ".part")
                with z.open(by_name[im]) as src, part.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(part, target)
    finally:
        zpath.unlink(missing_ok=True)
    return missing


def write_index(index: Path, drop: list[str]) -> None:
    """siclib's GT h5 without the images the release no longer has, written then renamed."""
    import h5py

    part = index.with_name(index.name + ".part")
    shutil.copy(GT_H5, part)
    with h5py.File(part, "a") as f:
        for key in drop:
            del f[key]
    os.replace(part, index)


def verify(out_dir: Path) -> None:
    """Every image decodes at the (h, w) its ground truth says -- the check AnyCalib never makes."""
    import cv2
    import h5py

    bad = []
    with h5py.File(out_dir / "images.h5", "r") as f:
        for key in f.keys():
            image = cv2.imread(str(out_dir / "images" / key), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            h, w = int(f[key].attrs["h"]), int(f[key].attrs["w"])
            if image is None or image.shape[:2] != (h, w):
                bad.append((key, None if image is None else image.shape[:2], (h, w)))
    assert not bad, f"{len(bad)} images do not match their GT size, e.g. {bad[:3]}"


def prepare(args) -> None:
    token = read_token(args)
    check_token(token)
    scenes = benchmark_images(args.limit)
    out_dir = Path(args.out) / NAME
    missing_txt = out_dir / "images_missing.txt"
    known = set(missing_txt.read_text(encoding="utf-8").split()) if missing_txt.exists() else set()
    missing = sorted(known)
    scenes = {s: [im for im in ims if f"{s}_{im}" not in known] for s, ims in scenes.items()}
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    cache = influx.cache_dir(out_dir / "_cache", args.cache)
    n_img = sum(len(v) for v in scenes.values())
    print(f"{NAME}: {n_img} images from {len(scenes)} scenes -> {img_dir} (zips in {cache})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {s: pool.submit(fetch_scene, s, ims, token, img_dir, cache) for s, ims in scenes.items()}
        for i, (scene, fut) in enumerate(futures.items(), 1):
            gone = fut.result()
            missing += [f"{scene}_{im}" for im in gone]
            note = f", {len(gone)} not in the release: {gone}" if gone else ""
            print(f"  [{i}/{len(scenes)}] {scene}: {len(scenes[scene])} images{note}", flush=True)
    missing = sorted(set(missing))
    missing_txt.write_text("".join(f"{m}\n" for m in missing), encoding="utf-8")

    if args.limit:
        print(f"{NAME}: --limit run, images.h5 not written", flush=True)
        return
    write_index(out_dir / "images.h5", drop=missing)
    verify(out_dir)
    shutil.rmtree(out_dir / "_cache", ignore_errors=True)
    print(f"{NAME}: complete, {n_img - len(missing)} images verified against images.h5, "
          f"{len(missing)} not in the release (images_missing.txt) -> {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None, help="benchmark root (default: paths config)")
    p.add_argument("--token_file", type=Path, default=None,
                   help="file holding the ScanNet++ token (default: scannetpp_token in the paths config)")
    p.add_argument("--cache", type=Path, default=None,
                   help="where the zips land (default: /work/<job id> in a job, else <out>/scannetpp2k/_cache)")
    p.add_argument("--paths", default=None, help="machine in configs/paths (default: auto)")
    p.add_argument("--workers", type=int, default=4,
                   help="scenes downloaded at once (the server rate-limits per IP; default: 4)")
    p.add_argument("--limit", type=int, default=None, help="first N scenes only (smoke test)")
    slurm.add_arguments(p, job_name="prepare-scannetpp", log_dir="logs/prepare_scannetpp/%t/%j", gpus=0,
                        partition="short", cpus_per_task=4, mem_gb=16, timeout_min=720)
    return p


def run(args) -> None:
    args.out = Path(args.out) if args.out else paths.get("benchmarks", args.paths)
    prepare(args)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.submitit:
        slurm.submit(args, argv, parser, module="prepare_scannetpp",
                     module_dir=Path(__file__).resolve().parent, tag=NAME)
        return
    run(args)


if __name__ == "__main__":
    main()

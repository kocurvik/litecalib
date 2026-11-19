"""MegaScenes -> real-image training views with COLMAP's own radial cameras."""

from __future__ import annotations

import argparse
import collections
import faulthandler
import hashlib
import html
import http.client
import json
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths
import slurm
import influx
import colmap_views as md

BUCKET = "https://megascenes.s3.us-west-2.amazonaws.com/"
NAME, CAM_ID = "megascenes_dist", "radial:1"
EXCLUDE_FILE = HERE / "megascenes_exclude.json"
META_COLUMNS = md.META_COLUMNS + ("license",)
CAMERA_BYTES = 24 + 8 * md.COLMAP_NUM_PARAMS[md.SIMPLE_RADIAL]
SPLITS = ("train", "val")
SCENE_ID = re.compile(r"^\d{6}$")


HOST = urllib.parse.urlsplit(BUCKET).netloc
_local = threading.local()


def connection(fresh: bool = False) -> http.client.HTTPSConnection:
    """This thread's connection to the bucket, reused across requests: a new TCP + TLS handshake per object
    (thousands a minute) is what a compute node's NAT cannot keep up with.
    """
    conn = getattr(_local, "conn", None)
    if conn is None or fresh:
        if conn is not None:
            conn.close()
        conn = _local.conn = http.client.HTTPSConnection(HOST, timeout=60)
    return conn


def fetch(url: str, retries: int = 5) -> bytes:
    """One object over HTTP; a transient failure is retried with a growing pause on a fresh connection, a 404 is
    not (raised as `urllib.error.HTTPError`).
    """
    path = url[len(BUCKET) - 1:]
    for attempt in range(retries):
        conn = connection(fresh=attempt > 0)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            body = response.read()
            if response.status == 200:
                return body
            if response.status == 404 or attempt == retries - 1:
                raise urllib.error.HTTPError(url, response.status, response.reason, response.headers, None)
        except (OSError, http.client.HTTPException):
            connection(fresh=True)
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def list_prefix(prefix: str, delimiter: str | None = "/") -> tuple[list[str], list[tuple[str, int]]]:
    """S3's ListObjectsV2 over anonymous HTTP: the common prefixes and the ``(key, size)`` under ``prefix``,
    across continuation pages.
    """
    prefixes, keys, token = [], [], None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            query["delimiter"] = delimiter
        if token:
            query["continuation-token"] = token
        page = fetch(BUCKET + "?" + urllib.parse.urlencode(query)).decode()
        prefixes += [html.unescape(p) for p in re.findall(r"<CommonPrefixes><Prefix>([^<]+)</Prefix>", page)]
        keys += [(html.unescape(k), int(s))
                 for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", page, re.S)]
        more = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", page)
        if not more:
            return prefixes, keys
        token = html.unescape(more.group(1))


def scene_dir(scene: str) -> str:
    """``000533`` -> ``000/533/``: the bucket's two-level folders."""
    assert SCENE_ID.match(scene), scene
    return f"{scene[:3]}/{scene[3:]}/"


def object_url(key: str) -> str:
    return BUCKET + urllib.parse.quote(key)


def reconstructed_scenes() -> list[str]:
    """Every scene with a reconstruction, from the listing (~460 requests)."""
    tops, _ = list_prefix("reconstruct/")
    scenes = []
    for top in tops:
        under, _ = list_prefix(top)
        scenes += [p.split("/")[1] + p.split("/")[2] for p in under]
    return sorted(scenes)


def scene_components(scene: str) -> list[tuple[str, int]]:
    """``(key prefix of a reconstruction, cameras in it)`` from the listing's sizes alone."""
    _, keys = list_prefix(f"reconstruct/{scene_dir(scene)}colmap/", delimiter=None)
    found = {}
    for key, size in keys:
        if key.endswith("/cameras.bin"):
            found[key[: -len("cameras.bin")]] = max(0, (size - 8) // CAMERA_BYTES)
    return sorted(found.items())


def licences(scene: str, names: list[str]) -> dict[str, str]:
    """``photo name -> Commons licence short name`` for the registered photos, from each subcategory's
    `raw_metadata.json` (``"?"`` where the file or the entry is missing).
    """
    out = {}
    for subcategory in sorted({n.split("/")[1] for n in names if n.startswith("commons/")}):
        try:
            raw = json.loads(fetch(object_url(f"images/{scene_dir(scene)}commons/{subcategory}/raw_metadata.json")))
        except (urllib.error.HTTPError, ValueError):
            continue
        for element in raw.get("elements", []):
            for page in element.values():
                title = page.get("title", "")
                info = (page.get("imageinfo") or [{}])[0].get("extmetadata", {})
                short = info.get("LicenseShortName", {}).get("value")
                if title.startswith("File:") and short:
                    out[title[len("File:"):].replace(" ", "_")] = short
    return {n: out.get(n.split("/")[-1], "?") for n in names}


def corner_distortion(record: dict) -> float:
    """``k1 r^2`` at the image corner: the relative radial shift `SIMPLE_RADIAL` gives it."""
    return record["k1"] * (0.5 * (record["h"] ** 2 + record["w"] ** 2) ** 0.5 / record["f"]) ** 2


def implausible_camera(record: dict, min_theta_max: float, max_corner_distortion: float) -> bool:
    """A view whose COLMAP camera is a failed self-calibration, not a lens."""
    return record["theta_max_deg"] < min_theta_max or abs(corner_distortion(record)) > max_corner_distortion


def split_of(scene: str, val_fraction: float) -> str:
    """A fixed split by a hash of the id: the first ``val_fraction`` of the hash range is val."""
    u = int(hashlib.sha1(scene.encode()).hexdigest()[:8], 16) / 2 ** 32
    return "val" if u < val_fraction else "train"


def load_exclusions(path: Path) -> dict[str, dict]:
    """``scene id -> {"name", "benchmark"}`` of the scenes no split may hold."""
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    bad = [s for s in table if not SCENE_ID.match(s)]
    assert not bad, f"{path}: not six-digit scene ids: {bad[:5]}"
    return table


def read_scenes(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    scenes = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    bad = [s for s in scenes if not SCENE_ID.match(s)]
    assert not bad, f"{path}: not six-digit scene ids: {bad[:5]}"
    return scenes


def fetch_view(scene: str, name: str, read: Callable[[], bytes], camera: md.Camera, args) -> dict | str:
    """`build_view` after the photo's download, in a worker thread."""
    try:
        blob = read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "missing_photo"
        raise
    return md.build_view(scene, name, blob, camera, args)


class RemoteScene(md.Scene):
    """`colmap_views.Scene` whose photos are fetched by the pool, not read in the main thread."""

    def _take(self, name: str, read: Callable[[], bytes]) -> None:
        if name in self.ignored:
            self.dropped["small_reconstruction"] += 1
        elif name not in self.cameras:
            self.dropped["unregistered"] += 1
        elif self.selected is not None and name not in self.selected:
            pass
        else:
            self._submit(fetch_view, self.name, name, read, self.cameras[name], self.args)


def scene_views(scene: str, components: list[tuple[str, int]], args, pool) -> tuple[list[dict] | None, collections.Counter]:
    """A scene's views: its reconstructions' model files, then the registered photos through the pool. None when
    no reconstruction could be read.
    """
    current = RemoteScene(scene, args, pool, skip_names=set())
    for prefix, _ in components:
        component = prefix.rstrip("/").split("/")[-1]
        for model_file in ("cameras.bin", "images.bin"):
            blob = fetch(object_url(prefix + model_file))
            current.feed(model_file, component, lambda b=blob: b)
    names = sorted({n for table in current.tables.values() for n in table.get("images", {})})
    for name in names:
        url = object_url(f"images/{scene_dir(scene)}{name}")
        current.feed("image", name, lambda u=url: fetch(u))
    records = current.finish()
    if records:
        held = licences(scene, [r["fname"].split("/", 1)[1] for r in records])
        for record in records:
            record["license"] = held[record["fname"].split("/", 1)[1]]
    return records, current.dropped


def listings(scenes: list[str], pool, window: int):
    """`(scene, components)` in the given order, with the listings running `window` scenes ahead in the pool: a
    listing takes ~0.6 s, and most scenes are settled by it alone.
    """
    queue, it = collections.deque(), iter(scenes)

    def fill():
        while pool is not None and len(queue) < window:
            scene = next(it, None)
            if scene is None:
                return
            queue.append((scene, pool.submit(scene_components, scene)))

    fill()
    while True:
        if queue:
            scene, future = queue.popleft()
            yield scene, future.result()
        else:
            scene = next(it, None)
            if scene is None:
                return
            yield scene, scene_components(scene)
        fill()


def prepare(args) -> None:
    import cv2

    import data

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {split: out_dir / f"{split}.h5" for split in SPLITS}
    if args.resume and all(data.store_is_complete(p) for p in index.values()):
        print(f"{args.name}: both stores exist and are complete", flush=True)
        return
    excluded = {s: v for s, v in load_exclusions(args.exclude_scenes).items() if "subcategories" not in v}
    scenes = read_scenes(args.scenes)
    if scenes is None:
        cache = out_dir / "scene_ids.json"
        if args.resume and cache.exists():
            scenes = json.loads(cache.read_text())
        else:
            scenes = reconstructed_scenes()
            cache.write_text(json.dumps(scenes))
    wanted = [s for s in scenes if s not in excluded]
    attrs = {"name": args.name, "source": BUCKET, "variant": "radial", "cam_id": args.cam_id,
             "short_side": args.short_side, "min_short_side": args.min_short_side,
             "per_scene": args.per_scene or 0, "min_component": args.min_component,
             "val_fraction": args.val_fraction, "jpeg_quality": args.jpeg_quality,
             "min_theta_max": args.min_theta_max, "max_corner_distortion": args.max_corner_distortion}
    print(f"{args.name} ({args.cam_id}): {len(wanted)} scenes wanted of {len(scenes)} reconstructed "
          f"({len(scenes) - len(wanted)} excluded as benchmark landmarks); val by hash under "
          f"{args.val_fraction}; reconstructions under {args.min_component} photos left out", flush=True)

    if args.workers > 1:
        cv2.setNumThreads(1)
    start, failed, dropped, stored, views = time.time(), [], collections.Counter(), 0, 0
    with ExitStack() as stack:
        writers = {split: stack.enter_context(data.StoreWriter(
            index[split], {**attrs, "split": split}, resume=args.resume)) for split in SPLITS}
        placed = {scene: split for split, writer in writers.items() for scene in writer.done}
        meta = influx.MetaCsv(out_dir / "views_meta.csv", META_COLUMNS,
                              keep=list(placed) if args.resume else None)
        stack.callback(meta.close)
        pool = stack.enter_context(ThreadPoolExecutor(args.workers)) if args.workers > 1 else None
        todo = [s for s in wanted if s not in placed]
        for scene, components in listings(todo, pool, window=4 * args.workers):
            if args.limit and stored >= args.limit:
                break
            split = split_of(scene, args.val_fraction)
            if sum(n for _, n in components) * args.prefilter_ratio < args.min_component:
                writers[split].add(scene, [])
                placed[scene] = split
                dropped["small_scene"] += 1
                if len(placed) % 1000 == 0:
                    print(f"  [{len(placed)}/{len(wanted)}] {stored} scenes stored, {views} views, "
                          f"{dropped['small_scene']} settled from the listing  "
                          f"[{(time.time() - start) / 60:.0f} min]", flush=True)
                continue
            records, reasons = scene_views(scene, components, args, pool)
            if records is None:
                print(f"  {scene}: no readable reconstruction - skipped", flush=True)
                failed.append(scene)
                continue
            kept = [r for r in records if not implausible_camera(r, args.min_theta_max, args.max_corner_distortion)]
            if len(kept) < len(records):
                reasons["implausible_camera"] += len(records) - len(kept)
            records = kept
            for record in records:
                record["split"] = split
            writers[split].add(scene, [(r, r["jpeg"]) for r in records])
            meta.add(records)
            placed[scene] = split
            dropped.update(reasons)
            stored += 1
            views += len(records)
            note = ", ".join(f"{n} {reason}" for reason, n in sorted(reasons.items()))
            print(f"  [{len(placed)}/{len(wanted)}] {scene} ({split}) -> {len(records)} views"
                  + (f" (dropped: {note})" if note else "")
                  + f"  [{views} views so far, {(time.time() - start) / 60:.0f} min]", flush=True)
        kept = {split: [s for s in writers[split].done] for split in SPLITS}
        counts = [writers[split].close({"scenes": json.dumps(kept[split])}) for split in SPLITS]
        entries = meta.read()
    (out_dir / "scenes.json").write_text(json.dumps(
        {"variant": "radial", "failed": failed, "excluded": sorted(set(scenes) & set(excluded)),
         "dropped": dict(dropped), **kept}, indent=1), encoding="utf-8")
    columns = ("theta_max_deg", "vfov_deg", "crop_frac", "resize_factor", "aspect_dev", "k1")
    with_views = {split: len({e["scene"] for e in entries if e["split"] == split}) for split in SPLITS}
    print(f"{args.name}: " + " / ".join(f"{c} {s}" for s, c in zip(SPLITS, counts))
          + f" views from {with_views['train']} / {with_views['val']} scenes -> {out_dir}\n"
          + influx.summarise(entries, columns)
          + "\nlicences: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(e["license"] for e in entries).most_common(8))
          + (f"\ndropped: {dict(dropped)}" if dropped else "")
          + (f"\n{len(failed)} scenes without a readable reconstruction: {failed}" if failed else ""),
          flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", type=Path, default=None,
                   help="a file of six-digit scene ids to prepare (default: every reconstructed scene)")
    p.add_argument("--exclude_scenes", type=Path, default=EXCLUDE_FILE,
                   help="scenes no split may hold (the benchmark landmarks); --exclude is SLURM's")
    p.add_argument("--val_fraction", type=float, default=0.01, help="share of the scenes held out as val, by hash")
    p.add_argument("--short_side", type=int, default=600, help="the stored views' short side (600x600 / 600x800 / 600x900)")
    p.add_argument("--min_short_side", type=int, default=480, help="a photo whose crop is shorter is dropped, not upsampled")
    p.add_argument("--per_scene", type=int, default=None, help="at most this many photos of a scene, evenly by name")
    p.add_argument("--min_component", type=int, default=20,
                   help="a reconstruction with fewer registered photos is left out")
    p.add_argument("--prefilter_ratio", type=float, default=4.0,
                   help="a scene is skipped from the listing alone when its cameras x this < --min_component "
                        "(photos share cameras; 2x is what the scenes read so far reach)")
    p.add_argument("--max_aspect_dev", type=float, default=0.005, help="|fy/fx - 1| beyond which a view is dropped")
    p.add_argument("--min_theta_max", type=float, default=1.0,
                   help="deg; a view whose half-diagonal field of view is narrower is dropped (`implausible_camera`)")
    p.add_argument("--max_corner_distortion", type=float, default=0.5,
                   help="|k1 r^2| at the corner beyond which a view is dropped (`implausible_camera`)")
    p.add_argument("--max_c_offset", type=float, default=0.5, help="px of `c` off the center beyond which a view is dropped")
    p.add_argument("--name", default=NAME, help="dataset directory name")
    p.add_argument("--out", type=Path, default=None, help="dataset root (default: paths config, megascenes_root)")
    p.add_argument("--paths", default=None, help="machine in configs/paths (default: auto)")
    p.add_argument("--jpeg_quality", type=int, default=influx.JPEG_QUALITY)
    p.add_argument("--workers", type=int, default=md.default_workers(),
                   help="threads downloading and decoding a scene's photos")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many scenes read past the listing (a smoke run)")
    p.add_argument("--no_resume", dest="resume", action="store_false",
                   help="start the stores afresh instead of continuing them")
    slurm.add_arguments(p, job_name="megascenes", log_dir="logs/megascenes/%t/%j", gpus=0,
                        partition="short", cpus_per_task=16, mem_gb=64, timeout_min=1440)
    return p


def run(args) -> None:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    args.cam_id = CAM_ID
    args.variant = "radial"
    args.out = Path(args.out) if args.out else paths.get("megascenes_root", args.paths)
    if args.short_side % 6:
        raise SystemExit(f"--short_side must be a multiple of 6 (4:3 and 3:2 both need it), got {args.short_side}")
    if not 0 <= args.val_fraction < 1:
        raise SystemExit(f"--val_fraction must be in [0, 1), got {args.val_fraction}")
    prepare(args)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.submitit:
        slurm.submit(args, argv, parser, module="prepare_megascenes",
                     module_dir=Path(__file__).resolve().parent, tag=args.name)
        return
    run(args)


if __name__ == "__main__":
    main()

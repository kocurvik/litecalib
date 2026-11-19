"""Render a view dataset from panoramas, with the camera parameters sampled as AnyCalib samples them."""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent import futures
from math import pi, tan
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "data_utils"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths
import slurm
import vendor

CONFIG_DIR = ROOT / "configs" / "generate"
SPLITS = ("train", "val", "test")
PANORAMA_ROOTS = {
    "openpano_v2": "openpano_root",
    "openpano_v2_gen": "openpano_root",
    "commons_gen": "commons_root",
    "commons_sa_gen": "commons_sa_root",
}
JPEG_QUALITY = {"perspective": 95, "gen": 75}


def load_config(name: str, overrides: list[str], machine: str | None = None):
    """The config in `configs/generate/`, with `key=value` overrides and `base_dir` filled in."""
    from omegaconf import OmegaConf

    conf = OmegaConf.merge(
        OmegaConf.load(CONFIG_DIR / f"{name}.yaml"), OmegaConf.from_dotlist(overrides)
    )
    if OmegaConf.is_missing(conf, "base_dir"):
        key = PANORAMA_ROOTS.get(name)
        if key is None:
            raise SystemExit(
                f"no panorama root known for {name!r}: pass base_dir=<dir> explicitly, "
                f"or add it to PANORAMA_ROOTS (known: {sorted(PANORAMA_ROOTS)})"
            )
        conf.base_dir = str(paths.get(key, machine))
    if not (Path(conf.base_dir) / "panoramas").is_dir():
        raise SystemExit(f"{conf.base_dir}/panoramas does not exist")
    return conf


def dataset_kind(conf) -> str:
    """"gen" for an `intrinsics` list, "perspective" for a `parameter_dists` block."""
    if "intrinsics" in conf:
        return "gen"
    if "parameter_dists" in conf:
        return "perspective"
    raise SystemExit(
        f"cannot tell what {conf.get('name')!r} is: it has neither `intrinsics` (a gen "
        "dataset) nor `parameter_dists` (a perspective one)"
    )


def allocation_workers(per_worker_gb: float) -> int:
    """Workers for the allocation this process runs in."""
    if "SLURM_JOB_ID" not in os.environ:
        return 1
    mem_mb = os.environ.get("SLURM_MEM_PER_NODE") or os.environ.get("SLURM_MEM_PER_CPU")
    mem_gb = int(mem_mb) / 1024 if mem_mb else 4 * per_worker_gb
    if "SLURM_MEM_PER_NODE" not in os.environ and "SLURM_MEM_PER_CPU" in os.environ:
        mem_gb *= int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    return default_workers(mem_gb, per_worker_gb)


def default_workers(mem_gb: float, per_worker_gb: float = 3.0) -> int:
    """As many workers as the allocation affords, memory first."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(cpus, int(mem_gb / per_worker_gb)))


def _seed(text: str) -> int:
    """siclib's own string -> seed hash (`DatasetGenerator.sample_value`)."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)


class ViewParams:
    """Every view's parameters for a panorama, drawn by siclib's own samplers."""

    def __init__(self, conf):
        self.kind = dataset_kind(conf)
        self.conf = conf
        if self.kind == "gen":
            from siclib.datasets.create_dataset_from_pano_rays import DatasetGenerator
        else:
            from siclib.datasets.create_dataset_from_pano import DatasetGenerator
        logging.getLogger(DatasetGenerator.__module__).setLevel(logging.WARNING)
        self.gen = DatasetGenerator(conf=conf)
        if self.kind == "perspective" and "k1_hat" in conf.parameter_dists:
            raise NotImplementedError("radial perspective datasets (k1_hat) are not generated here")

    def __call__(self, stem: str) -> list[dict]:
        """One dict per view: its rendering inputs plus the `record` its index stores."""
        return self._gen(stem) if self.kind == "gen" else self._perspective(stem)

    def _perspective(self, stem: str) -> list[dict]:
        import numpy as np
        import torch

        g, n = self.gen, self.conf.images_per_pano
        yaws = np.linspace(0, 2 * np.pi, n, endpoint=False)
        keys = [k for k in self.conf.parameter_dists if k != "shape"]
        values = {k: [g.sample_value(k, stem + k + str(y)) for y in yaws] for k in keys}
        h, w = (int(v) for v in g.sample_value("shape", stem + "shape"))
        views = []
        for i, yaw in enumerate(yaws):
            vfov = float(values["vfov"][i])
            f = 0.5 * h / tan(0.5 * vfov)
            record = {"fname": f"{stem}_{i}.jpg"}
            record |= {k: values[k][i].item() for k in keys}
            record |= {"height": h, "width": w}
            views.append({
                "cam_id": "pinhole", "params": torch.tensor((f, f, 0.5 * w, 0.5 * h)),
                "h": h, "w": w, "roll": record["roll"], "pitch": record["pitch"],
                "yaw": float(yaw), "vfov": vfov, "resize_factor": record["resize_factor"],
                "record": record,
            })
        return views

    def _gen(self, stem: str) -> list[dict]:
        import numpy as np
        import torch

        g, c, n = self.gen, self.conf, self.conf.images_per_pano
        h, w = (int(v) for v in c.im_size)
        yaws = torch.linspace(0, 2 * pi, n + 1)[:-1]
        cam_ids = np.random.default_rng(_seed(f"{stem}cam_id")).choice(
            g.cam_ids, size=n, p=g.cam_weights
        )
        views = []
        for i, cam_id in enumerate(cam_ids):
            cam_id = str(cam_id)
            res_fac = g.sample_value(c.resize_factor, f"{stem}resize_factor{i}")
            roll = g.sample_value(c.roll, f"{stem}roll{i}")
            pitch = g.sample_value(c.pitch, f"{stem}pitch{i}")
            params, vfov = g.get_safe_params(g.cams[cam_id], g.cam_specs[cam_id], (stem, i))
            record = {
                "fname": f"{stem}_{i}.jpg", "h": h, "w": w, "roll": roll, "pitch": pitch,
                "vfov": float(vfov), "resize_factor": res_fac, "cam_id": cam_id,
                "params": [float(p) for p in params.cpu().numpy().astype(np.float32)],
            }
            views.append({
                "cam_id": cam_id, "params": torch.tensor(record["params"], dtype=torch.float64),
                "h": h, "w": w, "roll": roll, "pitch": pitch, "yaw": float(yaws[i]),
                "vfov": float(vfov), "resize_factor": res_fac, "record": record,
            })
        return views


_WORKER: dict = {}


def _init_worker(conf_dict: dict) -> None:
    import cv2
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    conf = OmegaConf.create(conf_dict)
    _WORKER["params"] = ViewParams(conf)
    _WORKER["quality"] = int(conf.get("jpeg_quality") or JPEG_QUALITY[dataset_kind(conf)])


def render_panorama(pano_path: str) -> list[tuple[dict, bytes]]:
    """Render every view of one panorama; return each kept view's entry and encoded JPEG."""
    import cv2
    import numpy as np
    import torch

    import render

    views = _WORKER["params"](Path(pano_path).stem)
    image = cv2.imread(pano_path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"could not read {pano_path}")
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
    pano = torch.from_numpy(image).permute(2, 0, 1)
    H = pano.shape[-2]

    kept = []
    for view in views:
        h, w = view["h"], view["w"]
        lon, lat = render.view_rays(view["cam_id"], view["params"], h, w,
                                    view["roll"], view["pitch"], view["yaw"])
        scale = pi / view["vfov"] * h / H * view["resize_factor"]
        img = render.sample_view(pano, lon, lat, scale)
        if float((img.sum(0) == 0).float().mean()) >= 0.01:
            continue
        bgr = np.ascontiguousarray(render.to_uint8(img).numpy()[..., ::-1])
        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _WORKER["quality"]])
        if not ok:
            raise IOError(f"could not encode {view['record']['fname']}")
        entry = {
            "fname": view["record"]["fname"], "cam_id": view["cam_id"], "h": h, "w": w,
            "params": [float(p) for p in view["params"].numpy().astype(np.float32)],
            "roll": float(view["roll"]), "pitch": float(view["pitch"]),
            "vfov": float(view["vfov"]), "resize_factor": float(view["resize_factor"]),
        }
        kept.append((entry, jpeg.tobytes()))
    return kept


def generate_split(conf, split: str, n_workers: int, resume: bool, limit: int | None) -> int | None:
    """Render one split into `{split}.h5`; returns the number of views, or None if skipped."""
    import numpy as np
    from omegaconf import OmegaConf

    import data

    out_root = Path(conf.base_dir) / conf.name
    index = out_root / f"{split}.h5"
    journal = out_root / f"{split}.partial.jsonl"
    blob = out_root / f"{split}.partial.jpeg"
    pano_dir = Path(conf.base_dir) / "panoramas" / split
    panoramas = []
    if pano_dir.is_dir():
        panoramas = sorted(p for p in pano_dir.glob("*") if p.is_file() and not p.name.startswith("."))
    if limit:
        panoramas = panoramas[:limit]
    if not panoramas:
        print(f"{split}: no panoramas in {pano_dir} - skipped", flush=True)
        return None
    if resume and index.exists() and not journal.exists():
        print(f"{split}: {index.name} exists - already complete", flush=True)
        return None

    done: dict[str, list] = {}
    if resume and journal.exists() and blob.exists():
        size = blob.stat().st_size
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(v["offset"] + v["length"] <= size for v in entry["views"]):
                done[entry["pano"]] = entry["views"]
    else:
        journal.unlink(missing_ok=True)
        blob.unlink(missing_ok=True)
    todo = [p for p in panoramas if p.stem not in done]
    print(f"{split}: {len(panoramas)} panoramas, {len(done)} already done, {len(todo)} to render "
          f"with {n_workers} worker(s)", flush=True)

    start = last = time.time()
    pool = futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker,
        initargs=(OmegaConf.to_container(conf, resolve=True),),
    )
    with blob.open("ab") as out, journal.open("a", encoding="utf-8") as log, pool:
        jobs = {pool.submit(render_panorama, str(p)): p.stem for p in todo}
        for count, job in enumerate(futures.as_completed(jobs), 1):
            stem = jobs[job]
            views = []
            out.seek(0, os.SEEK_END)
            for entry, jpeg in job.result():
                views.append({**entry, "offset": out.tell(), "length": len(jpeg)})
                out.write(jpeg)
            out.flush()
            done[stem] = views
            log.write(json.dumps({"pano": stem, "views": views}) + "\n")
            log.flush()
            now = time.time()
            if now - last > 60 or count == len(todo):
                rate = count / (now - start)
                print(f"  {split}: {len(done)}/{len(panoramas)} panoramas, {rate:.2f}/s, "
                      f"eta {(len(todo) - count) / rate / 60:.1f} min", flush=True)
                last = now

    entries = [v for p in panoramas for v in done[p.stem]]
    raw = np.fromfile(blob, dtype=np.uint8)

    def read(i: int):
        return raw[entries[i]["offset"] : entries[i]["offset"] + entries[i]["length"]]

    data.write_store(index, entries, read, attrs={
        "name": conf.name, "kind": dataset_kind(conf), "split": split,
        "config": OmegaConf.to_yaml(conf),
    })
    del raw
    journal.unlink()
    blob.unlink()
    print(f"{split}: {len(entries)} views from {len(panoramas)} panoramas -> {index}", flush=True)
    return len(entries)


def run(args) -> None:
    from omegaconf import OmegaConf

    conf = load_config(args.config, args.overrides, args.paths)
    kind = dataset_kind(conf)
    n_workers = args.workers or allocation_workers(args.workers_gb)
    out_root = Path(conf.base_dir) / conf.name
    print(OmegaConf.to_yaml(conf), flush=True)
    print(f"kind: {kind}, output: {out_root}, workers: {n_workers}", flush=True)
    if args.dry_run:
        return
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise SystemExit(
            f"{out_root} already has content. Move it aside for a fresh run (a fresh run over "
            "an old one would leave stale views next to the new ones), or pass --resume."
        )
    ViewParams(conf)
    out_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, out_root / "config.yaml")
    for split in SPLITS:
        generate_split(conf, split, n_workers, args.resume, args.limit)
    print(
        "done. Next: `python data_utils/check_views.py --data <name>`, and for Commons copy "
        "`attribution.csv` next to the views -- CC BY requires it to travel with them.",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config", help=f"a config name in {CONFIG_DIR.relative_to(ROOT)}")
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    p.add_argument("--paths", default=None, help="machine in configs/paths (default: auto)")
    p.add_argument("--resume", action="store_true",
                   help="continue a killed run: skip finished splits and finished panoramas")
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N panoramas per split (smoke test)")
    p.add_argument("--workers", type=int, default=None,
                   help="worker processes (default: from the SLURM allocation, see "
                        "--workers_gb; 1 outside SLURM)")
    p.add_argument("--workers_gb", type=float, default=3.0,
                   help="RAM to budget per worker; raise --mem_gb rather than lowering this")
    p.add_argument("--dry_run", action="store_true", help="print the config and stop")
    slurm.add_arguments(p, job_name="genviews", log_dir="logs/generate/%t/%j", gpus=0,
                        partition="short", cpus_per_task=64, mem_gb=250, timeout_min=120)
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.submitit:
        slurm.submit(args, argv, parser, module="generate_views",
                     module_dir=Path(__file__).resolve().parent, tag=args.config)
        return
    run(args)


if __name__ == "__main__":
    main()

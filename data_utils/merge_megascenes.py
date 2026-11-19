"""Merge MegaScenes stores prepared in parts into one."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

import paths
import influx
import prepare_megascenes as ms

FLOAT_META = ("h", "w", "f", "k1", "theta_max_deg")


def read_meta(path: Path) -> dict[str, list[dict]]:
    """`views_meta.csv` rows by split, in file order."""
    rows = collections.defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["split"]].append(row)
    return rows


def copy_split(source: Path, rows: list[dict], writer, meta, args, dropped: collections.Counter,
               excluded: set[str]) -> tuple[int, int]:
    """Append one part's split to the merged writer. Returns (views in, views kept)."""
    import h5py

    import data

    with h5py.File(source, "r") as f:
        data.check_complete(f, source)
        n = int(f.attrs["n"])
        keys = json.loads(f.attrs["keys"])
        fname = f["fname"].asstr()[...]
        assert len(fname) == n == len(rows), f"{source}: {len(fname)} rows, attrs n={n}, {len(rows)} csv rows"
        assert all(r["fname"] == x for r, x in zip(rows, fname)), f"{source}: views_meta.csv out of step with the store"
        cam_id = f["cam_id"].asstr()[...]
        h, w, nparams = f["h"][...], f["w"][...], f["nparams"][...]
        offset, length, params = f["offset"][...], f["length"][...], f["params"][...]
        extra = {c: f[c][...] for c in data.STORE_COLUMNS if len(f[c]) == n}
        scene_of = np.array([x.split("/", 1)[0] for x in fname])
        starts = {}
        for i, s in enumerate(scene_of):
            starts.setdefault(s, i)
        assert set(starts) <= set(keys), f"{source}: views of scenes the store does not list"
        kept_total = 0
        for k, key in enumerate(keys):
            if key in writer.done:
                raise SystemExit(f"{source}: scene {key} is already in an earlier part")
            if key in excluded:
                dropped["benchmark_landmark"] += int((scene_of == key).sum())
                continue
            if key not in starts:
                writer.add(key, [])
                continue
            lo = starts[key]
            hi = lo
            while hi < n and scene_of[hi] == key:
                hi += 1
            assert key not in scene_of[hi:hi + 1] and (scene_of[lo:hi] == key).all()
            blob = f["jpeg"][offset[lo]:offset[hi - 1] + length[hi - 1]]
            items, kept_rows = [], []
            for i in range(lo, hi):
                row = rows[i]
                record = {c: float(row[c]) for c in FLOAT_META}
                if ms.implausible_camera(record, args.min_theta_max, args.max_corner_distortion):
                    dropped["implausible_camera"] += 1
                    continue
                entry = {"fname": fname[i], "cam_id": cam_id[i], "h": int(h[i]), "w": int(w[i]),
                         "params": params[i, : nparams[i]].tolist(), **{c: float(v[i]) for c, v in extra.items()}}
                a = offset[i] - offset[lo]
                items.append((entry, blob[a:a + length[i]].tobytes()))
                kept_rows.append(row)
            writer.add(key, items)
            meta.add(kept_rows)
            kept_total += len(items)
            if k % 2000 == 0:
                print(f"    {source.parent.name}/{source.name}: {k}/{len(keys)} scenes", flush=True)
    return n, kept_total


def _keys(part: Path) -> list[str]:
    import h5py

    keys = []
    for split in ms.SPLITS:
        with h5py.File(part / f"{split}.h5", "r") as f:
            keys += json.loads(f.attrs["keys"])
    return keys


def merge(args) -> None:
    import data

    root = Path(args.root) if args.root else paths.get("megascenes_root", args.paths)
    parts = [root / p for p in args.parts]
    out = root / args.name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    metas = {p: read_meta(p / "views_meta.csv") for p in parts}
    excluded = {s for s, v in ms.load_exclusions(args.exclude_scenes).items() if "subcategories" not in v}
    with (parts[0] / "views_meta.csv").open(newline="", encoding="utf-8") as f:
        columns = next(csv.reader(f))
    meta = influx.MetaCsv(out / "views_meta.csv", columns)
    dropped, totals, lists = collections.Counter(), {}, {}
    for split in ms.SPLITS:
        import h5py

        with h5py.File(parts[0] / f"{split}.h5", "r") as f:
            attrs = {k: v for k, v in f.attrs.items() if k not in ("n", "nbytes", "keys", "complete", "format", "scenes")}
        attrs |= {"merged_from": json.dumps(args.parts), "min_theta_max": args.min_theta_max,
                  "max_corner_distortion": args.max_corner_distortion}
        with data.StoreWriter(out / f"{split}.h5", attrs, resume=False) as writer:
            seen = kept = 0
            for p in parts:
                a, b = copy_split(p / f"{split}.h5", metas[p].get(split, []), writer, meta, args, dropped, excluded)
                seen, kept = seen + a, kept + b
                print(f"  {p.name} {split}: {b} / {a} views kept  [{(time.time() - start) / 60:.0f} min]", flush=True)
            lists[split] = list(writer.done)
            writer.close({"scenes": json.dumps(lists[split])})
        totals[split] = (seen, kept)
    meta.close()
    summary = {"variant": "radial", "failed": [], "excluded": [], "dropped": collections.Counter()}
    for p in parts:
        part = json.loads((p / "scenes.json").read_text())
        summary["failed"] += part.get("failed", [])
        summary["excluded"] = sorted(set(summary["excluded"]) | set(part.get("excluded", [])))
        summary["dropped"].update(part.get("dropped", {}))
    summary["excluded"] = sorted(set(summary["excluded"]) | (excluded & {k for p in parts for k in _keys(p)}))
    summary["dropped"].update(dropped)
    summary = {**summary, "dropped": dict(summary["dropped"]), "merged_from": args.parts, **lists}
    (out / "scenes.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{args.name}: " + " / ".join(f"{totals[s][1]} {s}" for s in ms.SPLITS) + " views ("
          + " / ".join(f"{totals[s][0] - totals[s][1]}" for s in ms.SPLITS) + f" dropped: {dict(dropped)}) from "
          + " / ".join(f"{len(lists[s])}" for s in ms.SPLITS) + f" scenes -> {out}  [{(time.time() - start) / 60:.0f} min]",
          flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", nargs="+", required=True, help="store directories under --root, in order")
    p.add_argument("--name", required=True, help="the merged directory, created under --root; must not exist")
    p.add_argument("--root", type=Path, default=None, help="default: paths config, megascenes_root")
    p.add_argument("--paths", default=None, help="machine in configs/paths (default: auto)")
    defaults = ms.build_parser().parse_args([])
    p.add_argument("--exclude_scenes", type=Path, default=ms.EXCLUDE_FILE, help="scenes left out (benchmark landmarks)")
    p.add_argument("--min_theta_max", type=float, default=defaults.min_theta_max)
    p.add_argument("--max_corner_distortion", type=float, default=defaults.max_corner_distortion)
    return p


def main(argv: list[str] | None = None) -> None:
    merge(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

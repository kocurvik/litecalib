"""Shared plumbing of the data preparers: resumable HTTP, streaming, even sampling, intrinsics rescaling and the
resumable journal of written items."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HF_REAL = "princeton-vl/InFlux-Real"
HF_SYNTH = "princeton-vl/InFlux-Synth"

JPEG_QUALITY = 95

WORK_ROOT = Path("/work")


def hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _session():
    import requests

    session = requests.Session()
    token = os.environ.get("HF_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def download(url: str, dst: Path, retries: int = 5, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dst``, resuming a partial file with a Range request."""
    dst = Path(dst)
    if dst.exists():
        return dst
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    session = _session()
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=120) as r:
                if have and r.status_code == 200:
                    part.unlink()
                    have = 0
                elif have and r.status_code == 416:
                    break
                r.raise_for_status()
                with part.open("ab" if have else "wb") as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
            break
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            print(f"  {url}: {type(exc).__name__}: {exc} - retrying in {wait}s", flush=True)
            time.sleep(wait)
    os.replace(part, dst)
    return dst


def cache_dir(fallback: Path, cache: Path | str | None = None) -> Path:
    """Where a run keeps the bytes nothing outside it will ever read."""
    if cache is not None:
        path = Path(cache)
    else:
        job = os.environ.get("SLURM_JOB_ID")
        path = WORK_ROOT / job if job else Path(fallback)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"  {path}: {type(exc).__name__}: {exc} - falling back to {fallback}", flush=True)
        path = Path(fallback)
        path.mkdir(parents=True, exist_ok=True)
    return path


def even_sample(candidates: Sequence[int], n: int) -> list[int]:
    """``n`` evenly spaced values from the sorted ``candidates``, or all of them if fewer."""
    candidates = list(candidates)
    if n <= 0 or len(candidates) <= n:
        return candidates
    pos = np.unique(np.round(np.linspace(0, len(candidates) - 1, n)).astype(int))
    return [candidates[i] for i in pos]


def scale_params(cam_id: str, params: np.ndarray, scale: float) -> np.ndarray:
    """Intrinsics of the same camera after the image is resized by ``scale``."""
    import torch

    from litecalib.anycalib.cameras import CameraFactory

    cam = CameraFactory.create_from_id(cam_id)
    p = torch.as_tensor(np.asarray(params, dtype=np.float64))
    out = cam.scale_and_shift(p, torch.full((2,), float(scale), dtype=p.dtype), torch.zeros(2, dtype=p.dtype))
    return out.numpy()


def max_tangential_px(params: np.ndarray, p1: float, p2: float, h: int, w: int, n: int = 9) -> float:
    """Largest pixel displacement the Brown-Conrady tangential terms produce over the image."""
    fx, fy, cx, cy = (float(v) for v in params[:4])
    gy, gx = np.meshgrid(np.linspace(-cy, h - cy, n), np.linspace(-cx, w - cx, n), indexing="ij")
    x, y = gx / fx, gy / fy
    r2 = x * x + y * y
    dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return float(np.hypot(dx * fx, dy * fy).max())


class ShardLog:
    """Append-only ``(blob, jsonl)`` journal of finished items, for a resumable run."""

    def __init__(self, blob: Path, journal: Path, resume: bool = True):
        self.blob, self.journal = Path(blob), Path(journal)
        self.done: dict[str, list[dict]] = {}
        if resume and self.journal.exists() and self.blob.exists():
            size = self.blob.stat().st_size
            for line in self.journal.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if all(v["offset"] + v["length"] <= size for v in record["entries"]):
                    self.done[record["key"]] = record["entries"]
        else:
            self.journal.unlink(missing_ok=True)
            self.blob.unlink(missing_ok=True)
        self.blob.parent.mkdir(parents=True, exist_ok=True)
        self._out = self.blob.open("ab")
        self._log = self.journal.open("a", encoding="utf-8")

    def add(self, key: str, items: Iterable[tuple[dict, bytes]]) -> list[dict]:
        entries = []
        self._out.seek(0, os.SEEK_END)
        for entry, jpeg in items:
            entries.append({**entry, "offset": self._out.tell(), "length": len(jpeg)})
            self._out.write(jpeg)
        self._out.flush()
        self.done[key] = entries
        self._log.write(json.dumps({"key": key, "entries": entries}) + "\n")
        self._log.flush()
        return entries

    def close(self) -> None:
        if not self._out.closed:
            self._out.close()
        if not self._log.closed:
            self._log.close()

    def write_stores(
        self, specs: Sequence[tuple[Path, Sequence[str], dict[str, Any]]], cleanup: bool = True
    ) -> list[int]:
        """Assemble one store per ``(index, key order, attrs)``, then drop the journal."""
        import data

        self.close()
        raw = np.fromfile(self.blob, dtype=np.uint8)
        counts = []
        for index, order, attrs in specs:
            entries = [e for key in order for e in self.done.get(key, [])]

            def read(i: int, _entries=entries):
                return raw[_entries[i]["offset"] : _entries[i]["offset"] + _entries[i]["length"]]

            data.write_store(Path(index), entries, read, attrs=attrs)
            counts.append(len(entries))
        del raw
        if cleanup:
            self.journal.unlink(missing_ok=True)
            self.blob.unlink(missing_ok=True)
        return counts

    def write_store(self, index: Path, order: Sequence[str], attrs: dict[str, Any]) -> int:
        """Assemble a single store in ``order`` and drop the journal. Returns the view count."""
        return self.write_stores([(Path(index), order, attrs)])[0]


class MetaCsv:
    """The per-sample sidecar, written as the run goes and kept in step with the store."""

    def __init__(self, path: Path, columns: Sequence[str], keep: Sequence[str] | None = None,
                 key: str = "scene"):
        import csv

        self.path, self.columns, self.key = Path(path), list(columns), key
        rows = []
        if keep is not None and self.path.exists():
            keep = set(keep)
            with self.path.open(newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r.get(key) in keep]
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns, extrasaction="ignore")
        self._writer.writeheader()
        self.add(rows)

    def add(self, rows: Sequence[dict]) -> None:
        self._writer.writerows(rows)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def read(self) -> list[dict]:
        """Everything written so far, for the run's closing summary."""
        import csv

        self.close()
        with self.path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


def write_meta_csv(path: Path, entries: Sequence[dict], columns: Sequence[str]) -> None:
    """The per-sample sidecar: everything the store has no column for."""
    import csv

    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)


def _as_float(value) -> float:
    """A number, or NaN for the missing values and empty csv cells a variant leaves behind."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarise(entries: Sequence[dict], keys: Sequence[str]) -> str:
    """A one-line percentile summary per key, for the run's log."""
    lines = []
    for key in keys:
        v = np.array([_as_float(e.get(key)) for e in entries], dtype=np.float64)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        pct = np.percentile(v, [0, 5, 50, 95, 100])
        lines.append(f"  {key:16s} min {pct[0]:9.3f}  p5 {pct[1]:9.3f}  med {pct[2]:9.3f}  "
                     f"p95 {pct[3]:9.3f}  max {pct[4]:9.3f}")
    return "\n".join(lines)

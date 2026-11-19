"""Training and evaluation data: view stores, benchmark indices, knot ground truth, sampling and loaders."""

import json
import logging
import os
import random
from math import ceil, gcd, sqrt
from pathlib import Path
from typing import Any, NamedTuple, Sequence

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, get_worker_info

import gt
import vendor
from litecalib import geometry
from litecalib.anycalib.cameras import CameraFactory
from litecalib.anycalib.cameras.base import BaseCamera

logger = logging.getLogger(__name__)

MAX_NPARAMS = BaseCamera.MAX_NPARAMS

__all__ = [
    "augmentation_mismatches",
    "centered_resize",
    "crop_resize_plan",
    "index_files",
    "load_index",
    "read_bytes",
    "read_image",
    "resize_plan",
    "scratch_dir",
    "write_store",
    "BatchedRandomSampler",
    "DataPoint",
    "KnotDataset",
    "MergedDataset",
    "ViewStore",
    "build_dataset",
    "make_loader",
]


ANYCALIB_AUG_INTENT = {
    "ImageCompression": ("quality", (20, 100), ("quality_range", ("quality_lower", "quality_upper"))),
    "Downscale": ("scale", (0.5, 0.99), ("scale_range", ("scale_min", "scale_max"))),
}
_AUG_WARNED = False


def _walk_transforms(transforms):
    """Flatten albumentations containers (Compose, OneOf, OneOrOther, ...)."""
    for t in transforms:
        inner = getattr(t, "transforms", None)
        if inner:
            yield from _walk_transforms(inner)
        else:
            yield t
            for attr in ("first", "second"):
                if (branch := getattr(t, attr, None)) is not None:
                    yield from _walk_transforms(getattr(branch, "transforms", [branch]))


def augmentation_mismatches(name: str = "geocalib") -> list[str]:
    """Parameters of AnyCalib's augmentations that the installed albumentations ignored."""
    if name != "geocalib":
        return []
    from siclib.datasets.augmentations import augmentations as siclib_augmentations

    aug = siclib_augmentations[name](OmegaConf.create({"name": name, "p": 1.0, "verbose": False}))
    out = []
    for t in _walk_transforms(aug.transforms):
        spec = ANYCALIB_AUG_INTENT.get(type(t).__name__)
        if spec is None:
            continue
        label, intended, (new_key, old_keys) = spec
        params = t.to_dict()["transform"]
        got = params.get(new_key)
        if got is None and all(k in params for k in old_keys):
            got = tuple(params[k] for k in old_keys)
        if got is None or tuple(got) != intended:
            out.append(
                f"{type(t).__name__}: AnyCalib asks for {label} {intended}, "
                f"this albumentations applies {tuple(got) if got else 'unknown'}"
            )
    return list(dict.fromkeys(out))


def warn_on_augmentation_mismatch(name: str) -> list[str]:
    """Log `augmentation_mismatches` once per process."""
    global _AUG_WARNED
    mismatches = augmentation_mismatches(name)
    if mismatches and not _AUG_WARNED:
        _AUG_WARNED = True
        logger.warning(
            "the installed albumentations ignores %d of AnyCalib's augmentation settings:\n  %s\n"
            "  -> pin albumentations to 1.4.x to train with AnyCalib's actual recipe",
            len(mismatches), "\n  ".join(mismatches),
        )
    return mismatches


class DataPoint(NamedTuple):
    """One image: where its bytes are, its size, and its ground-truth camera."""

    name: str
    file_name: str
    h: int
    w: int
    cam_id: str
    params: np.ndarray
    store: str = ""
    index: int = -1


STORE_FORMAT = "radcalib-views/1"

MEGASCENES_EXCLUDE = Path(__file__).resolve().parent / "data_utils" / "megascenes_exclude.json"
_MEGASCENES_EXCLUDE: dict[str, dict] | None = None


def excluded_megascenes_view(name: str) -> bool:
    """Whether a MegaScenes view is listed in `MEGASCENES_EXCLUDE`."""
    global _MEGASCENES_EXCLUDE
    if _MEGASCENES_EXCLUDE is None:
        _MEGASCENES_EXCLUDE = json.loads(MEGASCENES_EXCLUDE.read_text(encoding="utf-8"))
    parts = name.split("/")
    entry = _MEGASCENES_EXCLUDE.get(parts[0])
    if entry is None:
        return False
    return "subcategories" not in entry or parts[2] in entry["subcategories"]


STORE_COLUMNS = ("roll", "pitch", "vfov", "resize_factor")


class ViewStore:
    """One split's views in one HDF5 file: every view's JPEG bytes concatenated into a single ``uint8`` array
    ``jpeg``, located by ``offset`` / ``length``, next to per-view ``fname``, ``cam_id``, ``h``, ``w``,
    ``params`` (padded to `MAX_NPARAMS`, real length ``nparams``) and the generator's `STORE_COLUMNS`.
    """

    def __init__(self, path: Path | str, preload: bool = False):
        import h5py

        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            if f.attrs.get("format") != STORE_FORMAT:
                raise ValueError(f"{path} is not a {STORE_FORMAT} store")
            check_complete(f, self.path)
            self.offset = f["offset"][...]
            self.length = f["length"][...]
            self.jpeg = f["jpeg"][...] if preload else None
        self._file = None
        self._pid = None

    @property
    def preloaded(self) -> bool:
        return self.jpeg is not None

    def read(self, i: int) -> np.ndarray:
        """The encoded JPEG of view ``i``, a 1-D uint8 array."""
        o, n = int(self.offset[i]), int(self.length[i])
        if self.jpeg is not None:
            return self.jpeg[o : o + n]
        if self._file is None or self._pid != os.getpid():
            import h5py

            self._file, self._pid = h5py.File(self.path, "r"), os.getpid()
        return self._file["jpeg"][o : o + n]

    def __getstate__(self):
        return {**self.__dict__, "_file": None, "_pid": None}


_STORES: dict[str, ViewStore] = {}


def get_store(path: str, preload: bool = False) -> ViewStore:
    """The process-wide `ViewStore` for ``path``, loading it into RAM if asked to."""
    store = _STORES.get(path)
    if store is None or (preload and not store.preloaded):
        store = _STORES[path] = ViewStore(path, preload)
    return store


def is_store(path: Path) -> bool:
    if Path(path).suffix != ".h5" or not Path(path).exists():
        return False
    import h5py

    with h5py.File(path, "r") as f:
        return f.attrs.get("format") == STORE_FORMAT


def check_complete(f, path) -> None:
    """Refuse a store a `StoreWriter` has not finished writing."""
    if not f.attrs.get("complete", True):
        raise ValueError(
            f"{path} is still being written ({int(f.attrs.get('n', 0))} views, "
            f"{len(json.loads(f.attrs.get('keys', '[]')))} items so far): a run is either "
            "still going or was killed, and resuming it is what finishes the store"
        )


def store_is_complete(path: Path) -> bool:
    """True for a store that is finished, False for a partial one or for no store at all."""
    import h5py

    if not is_store(path):
        return False
    with h5py.File(path, "r") as f:
        return bool(f.attrs.get("complete", True))


def scratch_dir() -> Path | None:
    """The job's node-local scratch, ``/work/<SLURM_JOB_ID>``, if this runs in a job that has one."""
    job = os.environ.get("SLURM_JOB_ID")
    if job and Path(f"/work/{job}").is_dir():
        return Path(f"/work/{job}")
    return None


STAGE_DIR = "stage"


def stage_store(directory: Path, split: str = "train", scratch: Path | None = None,
                timeout: float = 7200.0) -> Path:
    """Copy ``<directory>/<split>.h5`` to the job's node-local scratch and return the directory it is in there
    (same name, so the dataset keeps it), for a store that is read per view.
    """
    import shutil
    import time

    directory = Path(directory)
    src = directory / f"{split}.h5"
    scratch = scratch_dir() if scratch is None else Path(scratch)
    if scratch is None:
        print(f"  {directory.name}: no /work/<job id> here, read in place", flush=True)
        return directory
    if not is_store(src):
        print(f"  {directory.name}: {src.name} is not a view store, read in place", flush=True)
        return directory
    dst_dir = scratch / STAGE_DIR / directory.name
    dst = dst_dir / src.name
    skip = dst_dir / f"{src.name}.skip"
    if dst.exists():
        return dst_dir
    if skip.exists():
        return directory
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        deadline = time.time() + timeout
        while not (dst.exists() or skip.exists()):
            if time.time() > deadline:
                raise TimeoutError(f"waited {timeout:.0f} s for rank 0 to stage {src} to {dst_dir}")
            time.sleep(1.0)
        return directory if skip.exists() and not dst.exists() else dst_dir

    dst_dir.mkdir(parents=True, exist_ok=True)
    size, free = src.stat().st_size, shutil.disk_usage(scratch).free
    if free < size + (2 << 30):
        print(f"  {directory.name}: {size / 1e9:.1f} GB does not fit the {free / 1e9:.1f} GB free in "
              f"{scratch}, read in place", flush=True)
        skip.touch()
        return directory
    print(f"  staging {src} ({size / 1e9:.1f} GB) to {dst_dir} ...", flush=True)
    start = time.time()
    part = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, part)
    os.replace(part, dst)
    print(f"  staged {directory.name} in {time.time() - start:.0f} s", flush=True)
    return dst_dir


def clear_stage() -> None:
    """Remove what `stage_store` copied to this job's scratch (a no-op outside a job)."""
    import shutil

    scratch = scratch_dir()
    if scratch is not None:
        shutil.rmtree(scratch / STAGE_DIR, ignore_errors=True)


def write_store(path: Path, entries: list[dict], read, attrs: dict | None = None) -> None:
    """Write a `ViewStore`: ``entries`` in order, ``read(i)`` giving entry i's JPEG bytes."""
    import shutil

    import h5py

    path = Path(path)
    final_tmp = path.with_name(path.name + ".tmp")
    scratch = scratch_dir()
    tmp = scratch / f"{path.parent.name}_{path.name}" if scratch else final_tmp
    n = len(entries)
    lengths = np.array([e["length"] for e in entries], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64) if n else lengths
    params = np.zeros((n, MAX_NPARAMS), dtype=np.float32)
    for i, e in enumerate(entries):
        params[i, : len(e["params"])] = np.asarray(e["params"], dtype=np.float32)
    strings = h5py.string_dtype()
    with h5py.File(tmp, "w") as f:
        f.attrs["format"] = STORE_FORMAT
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
        f.create_dataset("fname", data=np.array([e["fname"] for e in entries], dtype=object), dtype=strings)
        f.create_dataset("cam_id", data=np.array([e["cam_id"] for e in entries], dtype=object), dtype=strings)
        f["h"] = np.array([e["h"] for e in entries], dtype=np.int32)
        f["w"] = np.array([e["w"] for e in entries], dtype=np.int32)
        f["params"] = params
        f["nparams"] = np.array([len(e["params"]) for e in entries], dtype=np.int8)
        for key in STORE_COLUMNS:
            if entries and key in entries[0]:
                f[key] = np.array([e[key] for e in entries], dtype=np.float64)
        f["offset"] = offsets
        f["length"] = lengths
        jpeg = f.create_dataset("jpeg", shape=(int(lengths.sum()),), dtype=np.uint8)
        for i in range(n):
            jpeg[offsets[i] : offsets[i] + lengths[i]] = read(i)
    if scratch:
        print(f"copying {tmp} to {path}", flush=True)
        shutil.copy(tmp, final_tmp)
        tmp.unlink()
    os.replace(final_tmp, path)


class StoreWriter:
    """A `ViewStore` written **incrementally**, one item at a time, and resumable."""

    JPEG_CHUNK = 4 << 20

    def __init__(self, path: Path | str, attrs: dict | None = None, resume: bool = True,
                 chunk: int = 4096):
        import h5py

        self.path = Path(path)
        self.attrs = dict(attrs or {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not (resume and self.path.exists() and is_store(self.path))
        if fresh:
            self.path.unlink(missing_ok=True)
        self._f = h5py.File(self.path, "w" if fresh else "a")
        strings = h5py.string_dtype()
        if fresh:
            self._f.attrs["format"] = STORE_FORMAT
            for key, value in self.attrs.items():
                self._f.attrs[key] = value
            for name, dtype in (("fname", strings), ("cam_id", strings), ("h", np.int32),
                                ("w", np.int32), ("nparams", np.int8), ("offset", np.int64),
                                ("length", np.int64), *((c, np.float64) for c in STORE_COLUMNS)):
                self._f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                       chunks=(chunk,))
            self._f.create_dataset("params", shape=(0, MAX_NPARAMS), maxshape=(None, MAX_NPARAMS),
                                   dtype=np.float32, chunks=(chunk, MAX_NPARAMS))
            self._f.create_dataset("jpeg", shape=(0,), maxshape=(None,), dtype=np.uint8,
                                   chunks=(self.JPEG_CHUNK,))
            self.done: list[str] = []
            self.n, self.nbytes = 0, 0
            self._commit()
        else:
            self.done = json.loads(self._f.attrs.get("keys", "[]"))
            self.n, self.nbytes = int(self._f.attrs.get("n", 0)), int(self._f.attrs.get("nbytes", 0))
            self._truncate()
            self._commit()

    @property
    def complete(self) -> bool:
        return bool(self._f.attrs.get("complete", False))

    def _truncate(self) -> None:
        for name in ("fname", "cam_id", "h", "w", "nparams", "offset", "length", "params",
                     *STORE_COLUMNS):
            dataset = self._f[name]
            if len(dataset) != self.n:
                dataset.resize(self.n, axis=0)
        if len(self._f["jpeg"]) != self.nbytes:
            self._f["jpeg"].resize((self.nbytes,))
        self._f.flush()

    def _commit(self, complete: bool = False) -> None:
        self._f.attrs["n"] = self.n
        self._f.attrs["nbytes"] = self.nbytes
        self._f.attrs["keys"] = json.dumps(self.done)
        self._f.attrs["complete"] = complete
        self._f.flush()

    def add(self, key: str, items: Sequence[tuple[dict, bytes]]) -> int:
        """Append one item's views (``(entry, jpeg bytes)``) and record it as done."""
        if key in self.done:
            return 0
        entries = [e for e, _ in items]
        blobs = [bytes(b) for _, b in items]
        count, total = len(entries), sum(len(b) for b in blobs)
        if count:
            lengths = np.array([len(b) for b in blobs], dtype=np.int64)
            offsets = self.nbytes + np.concatenate(([0], np.cumsum(lengths)[:-1]))
            params = np.zeros((count, MAX_NPARAMS), dtype=np.float32)
            for i, entry in enumerate(entries):
                params[i, : len(entry["params"])] = np.asarray(entry["params"], dtype=np.float32)
            columns = {
                "fname": np.array([e["fname"] for e in entries], dtype=object),
                "cam_id": np.array([e["cam_id"] for e in entries], dtype=object),
                "h": np.array([e["h"] for e in entries], dtype=np.int32),
                "w": np.array([e["w"] for e in entries], dtype=np.int32),
                "nparams": np.array([len(e["params"]) for e in entries], dtype=np.int8),
                "offset": offsets.astype(np.int64),
                "length": lengths,
                "params": params,
                **{c: np.array([e.get(c, 0.0) for e in entries], dtype=np.float64)
                   for c in STORE_COLUMNS if c in entries[0]},
            }
            for name, values in columns.items():
                dataset = self._f[name]
                dataset.resize(self.n + count, axis=0)
                dataset[self.n :] = values
            jpeg = self._f["jpeg"]
            jpeg.resize((self.nbytes + total,))
            jpeg[self.nbytes :] = np.frombuffer(b"".join(blobs), dtype=np.uint8)
            self._f.flush()
        self.n += count
        self.nbytes += total
        self.done.append(key)
        self._commit()
        return count

    def close(self, attrs: dict | None = None) -> int:
        """Mark the store finished (readable) and close it. Returns the view count."""
        if self._f:
            for key, value in {**self.attrs, **(attrs or {})}.items():
                self._f.attrs[key] = value
            self._commit(complete=True)
            self._f.close()
            self._f = None
        return self.n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._f and exc[0] is None:
            self.close()
        elif self._f:
            self._f.close()
            self._f = None
        return False


def _load_store(path: Path) -> list[DataPoint]:
    import h5py

    with h5py.File(path, "r") as f:
        check_complete(f, path)
        names = f["fname"].asstr()[...]
        cam_ids = f["cam_id"].asstr()[...]
        hs, ws = f["h"][...], f["w"][...]
        params, nparams = f["params"][...], f["nparams"][...]
    return [
        DataPoint(str(names[i]), "", int(hs[i]), int(ws[i]), str(cam_ids[i]),
                  params[i, : nparams[i]].copy(), str(path), i)
        for i in range(len(names))
    ]


def _load_anycalib_h5(h5_file: Path, img_root: Path) -> list[DataPoint]:
    """AnyCalib's HDF5 index (one group per image), as its `load_h5_anycalib_format` reads it."""
    import h5py

    with h5py.File(h5_file, "r") as f:
        return [
            DataPoint(
                name=name,
                file_name=str(img_root / name),
                h=int(group.attrs["h"]),
                w=int(group.attrs["w"]),
                cam_id=str(group.attrs["cam_id"]),
                params=group.attrs["params"].astype(np.float32),
            )
            for name, group in f.items()
        ]


def _load_anycalib_csv(csv_file: Path, img_root: Path) -> list[DataPoint]:
    """GeoCalib's CSV index, as AnyCalib's `load_csv_openpano_format` reads it."""
    import pandas as pd

    is_radial = "radial" in csv_file.parent.name
    cam_id = "radial:1" if is_radial else "pinhole"
    points = []
    for _, row in pd.read_csv(csv_file).iterrows():
        h, w = int(row["height"]), int(row["width"])
        f = row["focal"] if "focal" in row else h / 2 / np.tan(float(row["vfov"]) / 2)
        cx = row["px"] if "px" in row else 0.5 * w
        cy = row["py"] if "py" in row else 0.5 * h
        k1 = row.get("k1", None)
        assert not is_radial or k1 is not None, "k1 must be provided for radial cameras"
        params = np.array((f, f, cx, cy, *((k1,) if is_radial else ())), dtype=np.float32)
        points.append(DataPoint(str(row["fname"]), str(img_root / row["fname"]), h, w, cam_id, params))
    return points


def index_files(directory: Path, split: str | None = None) -> tuple[Path, Path] | None:
    """The (index, image directory) pair of a dataset, or None if there is none."""
    directory = Path(directory)
    stem, img_dir = ("images", directory / "images") if split is None else (split, directory / split)
    for suffix in (".h5", ".csv"):
        index = directory / f"{stem}{suffix}"
        if index.exists():
            return index, img_dir
    return None


def load_index(directory: Path, split: str | None = None) -> list[DataPoint]:
    """`DataPoint`s (name, bytes location, size, GT camera model and parameters), any layout."""
    found = index_files(directory, split)
    if found is None:
        wanted = "images.{h5,csv}" if split is None else f"{split}.{{h5,csv}}"
        raise FileNotFoundError(f"no {wanted} in {directory}")
    index, img_dir = found
    if index.suffix == ".csv":
        return _load_anycalib_csv(index, img_dir)
    return _load_store(index) if is_store(index) else _load_anycalib_h5(index, img_dir)


def read_bytes(dp: DataPoint) -> np.ndarray:
    """The encoded image of a datapoint, from its store or its file."""
    if dp.store:
        return get_store(dp.store).read(dp.index)
    return np.fromfile(dp.file_name, dtype=np.uint8)


def read_image_uint8(dp: DataPoint) -> np.ndarray:
    """(H, W, 3) uint8 RGB."""
    import cv2

    image = cv2.imdecode(read_bytes(dp), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"could not decode {dp.name} ({dp.store or dp.file_name})")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_image(dp: DataPoint) -> Tensor:
    """(3, H, W) float in [0, 1], as siclib's `load_image` returns it."""
    return torch.from_numpy(read_image_uint8(dp)).permute(2, 0, 1).float().div_(255)


def resize_plan(h: int, w: int, ht: int, wt: int, max_extra_upscale: float = 1.05) -> tuple:
    """Plan an isotropic resize + symmetric center crop from ``(h, w)`` to ``(ht, wt)``."""
    g = gcd(h, w)
    h_unit, w_unit = h // g, w // g
    n = max(ceil(ht / h_unit), ceil(wt / w_unit))
    minimal_scale = max(ht / h, wt / w)
    for n in range(n, n + 8):
        h1, w1 = n * h_unit, n * w_unit
        if h1 >= ht and w1 >= wt and (h1 - ht) % 2 == 0 and (w1 - wt) % 2 == 0:
            scale = n / g
            if scale > max_extra_upscale * max(minimal_scale, 1e-9):
                raise ValueError(
                    f"cannot resize {h}x{w} to {ht}x{wt} isotropically without upsampling by "
                    f"{scale:.3f} (vs {minimal_scale:.3f} needed): gcd({h}, {w}) = {g} is too "
                    "small. Generate views at a size sharing a large factor with the targets."
                )
            return scale, (h1, w1), ((h1 - ht) // 2, (w1 - wt) // 2)
    raise ValueError(f"no symmetric isotropic plan from {h}x{w} to {ht}x{wt}")


def crop_resize_plan(h: int, w: int, ht: int, wt: int) -> tuple[tuple[int, int, int, int], tuple[float, float]]:
    """Centre-crop to the target aspect ratio, then resize. Works for *any* size."""
    if w * ht > h * wt:
        crop_h, crop_w = h, round(h * wt / ht)
    else:
        crop_h, crop_w = round(w * ht / wt), w
    crop_h, crop_w = min(crop_h, h), min(crop_w, w)
    top, left = (h - crop_h) // 2, (w - crop_w) // 2
    return (top, left, crop_h, crop_w), (wt / crop_w, ht / crop_h)

def edit_region(h: int, w: int, crop_y: float, crop_x: float) -> tuple[int, int, int, int]:
    """The region an *edited* view keeps: AnyCalib's one-sided crop (`ImagePreprocessor`)."""
    cy, cx = int(crop_y * h), int(crop_x * w)
    top, left = max(-cy, 0), max(-cx, 0)
    return top, left, h - abs(cy), w - abs(cx)


def edited_resize(im: Tensor, region: tuple[int, int, int, int], target_size: tuple[int, int]
                  ) -> tuple[Tensor, Tensor, Tensor]:
    """Crop (3, H, W) to ``region`` and resize it to ``target_size`` without keeping the aspect ratio -- the CPU
    twin of what `train/gpu_pipeline.py` does with an edited plan.
    """
    top, left, hc, wc = region
    ht, wt = target_size
    im = im[..., top : top + hc, left : left + wc]
    if (hc, wc) != (ht, wt):
        im = F.interpolate(
            im[None], (ht, wt), mode="bicubic", align_corners=False, antialias=ht < hc or wt < wc
        )[0].clamp(0, 1)
    scale = torch.tensor((wt / wc, ht / hc))
    return im, scale, -torch.tensor((float(left), float(top))) * scale


def centered_resize(im: Tensor, target_size: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
    """Resize (3, H, W) to ``target_size`` keeping `c` at the image center and `fx == fy`."""
    h, w = im.shape[-2:]
    ht, wt = target_size
    if (h, w) == (ht, wt):
        return im, torch.ones(2), torch.zeros(2)
    scale, (h1, w1), (crop_h, crop_w) = resize_plan(h, w, ht, wt)
    if (h1, w1) != (h, w):
        im = F.interpolate(
            im[None], (h1, w1), mode="bicubic", align_corners=False, antialias=scale < 1
        )[0].clamp(0, 1)
    im = im[..., crop_h : h1 - crop_h, crop_w : w1 - crop_w]
    assert im.shape[-2:] == (ht, wt), f"{im.shape[-2:]} != {target_size}"
    return im, torch.full((2,), float(scale)), torch.tensor((-float(crop_w), -float(crop_h)))


def compute_target_size(
    ho: int, wo: int, res: float | None, ar: float | None, div: int | None
) -> tuple[int, int]:
    """The shape for a target resolution (H*W) and aspect ratio (H/W), each side a multiple of ``div`` --
    AnyCalib's `_SimpleDataset.compute_target_size`.
    """
    res = ho * wo if res is None else res
    ar = ho / wo if ar is None else ar
    w = sqrt(res / ar)
    h = ar * w
    return (int(h), int(w)) if div is None else (round(h / div) * div, round(w / div) * div)


def _from_unit(interval: tuple[float, float], rand: float) -> float:
    """A number in [0, 1) mapped into ``interval`` (AnyCalib's `map_rand_to_range`)."""
    return interval[0] + rand * (interval[1] - interval[0])


class KnotDataset(Dataset):
    """A view dataset with knot ground truth, sampled as AnyCalib samples."""

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        k: int = geometry.DEFAULT_K,
        placement: str = geometry.DEFAULT_PLACEMENT,
        flip_prob: float = 0.5,
        check_invariants: bool = True,
        limit: int | None = None,
        name: str | None = None,
        resolution: int | tuple[int, int] | None = 320**2,
        aspect_ratio: tuple[float, float] | None = (0.5, 2.0),
        edge_divisible_by: int | None = 14,
        augmentations: str = "geocalib",
        seed: int = 0,
        augment_after_resize: bool = True,
        jpeg_bytes: bool = False,
        to_closest_train_size: bool = False,
        preload: bool = False,
        profile: str = "radial",
        edit_prob: float = 0.0,
        edit_crop: float = 0.5,
        knot_targets: bool = True,
        field_repr: str = "fov",
    ):
        assert profile in ("radial", "field"), f"unknown profile {profile!r}"
        assert field_repr in geometry.FIELD_REPRS, f"unknown field representation {field_repr!r}"
        self.profile = profile
        self.field_repr = field_repr
        assert 0.0 <= edit_prob <= 1.0 and 0.0 <= edit_crop < 1.0, (edit_prob, edit_crop)
        assert not edit_prob or not knot_targets or profile == "field", (
            f"edited views have an off-center c and fy != fx, which the {profile!r} profile "
            "cannot represent; use --profile field"
        )
        self.edit_prob = edit_prob if split == "train" else 0.0
        self.edit_crop = edit_crop
        self.name = name or Path(dataset_dir).name
        self.split = split
        self.datapoints = load_index(Path(dataset_dir), split)[:limit]
        if Path(dataset_dir).name.startswith("megascenes"):
            n = len(self.datapoints)
            self.datapoints = [dp for dp in self.datapoints if not excluded_megascenes_view(dp.name)]
            print(f"{self.name} ({split}): {n - len(self.datapoints)} views of benchmark landmarks dropped", flush=True)
        for store in {dp.store for dp in self.datapoints if dp.store}:
            get_store(store, preload=preload)

        self.rng_tfs_order: list[str] = []
        if isinstance(resolution, (tuple, list)):
            assert len(resolution) == 2 and resolution[0] <= resolution[1], resolution
            self.res_pool: Any = tuple(resolution)
            self.rng_tfs_order.append("res")
        else:
            self.res_pool = resolution
        if aspect_ratio is None:
            self.ar_pool = None
        else:
            assert len(aspect_ratio) == 2 and aspect_ratio[0] <= aspect_ratio[1], aspect_ratio
            self.ar_pool = tuple(aspect_ratio)
            self.rng_tfs_order.append("ar")
        self.npools = len(self.rng_tfs_order)
        self.edge_divisible_by = edge_divisible_by
        self.to_closest_train_size = to_closest_train_size

        self.augmentation = None
        if split == "train" and not jpeg_bytes and augmentations != "identity":
            from siclib.datasets.augmentations import augmentations as registry

            self.augmentation = registry[augmentations](
                OmegaConf.create({"name": augmentations, "verbose": False})
            )
        self.augment_after_resize = augment_after_resize
        self.jpeg_bytes = jpeg_bytes
        self.k = k
        self.placement = placement
        self.flip_prob = flip_prob if split == "train" else 0.0
        self.check_invariants = check_invariants
        self.seed = seed

    def __len__(self) -> int:
        return len(self.datapoints)

    def _target_size(self, idx) -> tuple[DataPoint, tuple[int, int] | None]:
        """The datapoint and the shape it will have, as `_SimpleDataset.getitem` picks it."""
        if isinstance(idx, (tuple, list)):
            i, *rands = idx
            dp = self.datapoints[int(i)]
            rand = dict(zip(self.rng_tfs_order, rands))
            ar = _from_unit(self.ar_pool, rand["ar"]) if "ar" in rand else None
            res = _from_unit(self.res_pool, rand["res"]) if "res" in rand else self.res_pool
            return dp, compute_target_size(dp.h, dp.w, res, ar, self.edge_divisible_by)
        dp = self.datapoints[int(idx)]
        if not self.to_closest_train_size:
            return dp, None
        h, w = dp.h, dp.w
        ar = h / w
        if self.ar_pool is not None:
            ar = min(max(ar, self.ar_pool[0]), self.ar_pool[1])
        res: Any = self.res_pool
        if isinstance(res, tuple):
            res = h * w if res[0] <= h * w <= res[1] else sum(res) / 2
        return dp, compute_target_size(h, w, res, ar, self.edge_divisible_by)

    def _edit(self, idx, h: int, w: int) -> tuple[int, int, int, int] | None:
        """The region an edited training view keeps (`edit_region`), or None for a plain one."""
        if not self.edit_prob or not isinstance(idx, (tuple, list)):
            return None
        if not torch.rand(()) < self.edit_prob:
            return None
        crop_y, crop_x = ((torch.rand(2) * 2 - 1) * self.edit_crop).tolist()
        return edit_region(h, w, crop_y, crop_x)

    def _geometry(self, dp: DataPoint, ht: int, wt: int, params: Tensor,
                  edited: bool = False) -> dict[str, Any]:
        """Knots and flip for the camera as it is at ``(ht, wt)``."""
        if self.check_invariants and not edited:
            ok = gt.check_center_and_aspect(dp.cam_id, params.double(), ht, wt)
            assert bool(ok), (
                f"{self.name}/{dp.name}: the preprocessed camera is not centered / not square "
                f"at {ht}x{wt}: {params.tolist()}"
            )
        flip = 0
        if self.flip_prob and torch.rand(()) < self.flip_prob:
            flip = 1 if torch.rand(()) < 0.5 else 2
        if self.profile == "field":
            theta, theta_max, valid = self._field_knots(dp, params, ht, wt, flip)
        else:
            theta, theta_max, valid = gt.gt_knots(
                dp.cam_id, params.double(), ht, wt, self.k, self.placement, check_invariants=False
            )
        if edited and flip:
            params = params.clone()
            cam = CameraFactory.create_from_id(dp.cam_id)
            i = cam.NUM_F + (0 if flip == 1 else 1)
            params[i] = (wt if flip == 1 else ht) - params[i]
        padded = torch.cat((params, params.new_zeros(MAX_NPARAMS - len(params))))
        return {
            "name": dp.name,
            "dataset": self.name,
            "theta": theta.float(),
            "Theta": theta_max.float(),
            "valid": valid,
            "cam_id": dp.cam_id,
            "intrinsics": padded.float(),
            "nparams": len(params),
            "flip": flip,
        }

    def _field_knots(self, dp: DataPoint, params: Tensor, ht: int, wt: int, flip: int):
        """The field variant's target: `gt.gt_field_knots` (``2 K^2``, `geometry.field_values`), for the image
        *as flipped* (`geometry.field_flip`: mirrored, and the flipped axis's component negated -- a no-op for
        a centered camera, whose field is odd about the center, and right for any ``c``). ``Theta``
        (monitoring only) is the middle row's right-edge ``phi_x``: half the horizontal FoV when centered.
        """
        values, valid = gt.gt_field_knots(dp.cam_id, params.double(), ht, wt, self.k, self.field_repr)
        values = geometry.field_flip(values, self.k, flip)
        return values, values[(self.k // 2) * self.k + self.k - 1], valid

    def __getitem__(self, idx) -> dict[str, Any]:
        dp, target_size = self._target_size(idx)
        h, w = dp.h, dp.w
        ht, wt = (h, w) if target_size is None else target_size
        cam = CameraFactory.create_from_id(dp.cam_id)
        params = torch.as_tensor(dp.params, dtype=torch.float32)

        region = self._edit(idx, h, w)
        assert region is None or cam.NUM_F == 2, (
            f"{self.name}/{dp.name}: an edited view needs separate fx, fy; {dp.cam_id!r} has one"
        )
        if self.jpeg_bytes:
            if region is not None:
                top, left, hc, wc = region
                plan = (top, left, hc, wc, ht, wt, 0, 0)
                scale = torch.tensor((wt / wc, ht / hc))
                params = cam.scale_and_shift(
                    params, scale, -torch.tensor((float(left), float(top))) * scale, copy=True
                )
            elif (h, w) == (ht, wt):
                plan = (0, 0, h, w, h, w, 0, 0)
            else:
                scale, (h1, w1), (crop_h, crop_w) = resize_plan(h, w, ht, wt)
                plan = (0, 0, h, w, h1, w1, crop_h, crop_w)
                params = cam.scale_and_shift(
                    params, torch.full((2,), float(scale)),
                    torch.tensor((-float(crop_w), -float(crop_h))), copy=True,
                )
            sample = self._geometry(dp, ht, wt, params, edited=region is not None)
            sample["flip"] = torch.tensor(sample["flip"], dtype=torch.int8)
            return {
                **sample,
                "jpeg": torch.from_numpy(read_bytes(dp).copy()),
                "plan": torch.tensor(plan, dtype=torch.int32),
                "target_size": torch.tensor((ht, wt), dtype=torch.int32),
            }

        raw = read_image_uint8(dp)
        assert raw.shape[:2] == (h, w), f"{dp.name}: {raw.shape[:2]} != index {(h, w)}"
        before = self.augmentation is not None and not self.augment_after_resize
        if before:
            image = self.augmentation(raw, return_tensor=True)
        else:
            image = torch.from_numpy(raw).permute(2, 0, 1).float().div_(255)
        if region is not None:
            image, scale_xy, shift_xy = edited_resize(image, region, (ht, wt))
        else:
            image, scale_xy, shift_xy = centered_resize(image, (ht, wt))
        if self.augmentation is not None and not before:
            image = self.augmentation(image.permute(1, 2, 0).contiguous(), return_tensor=True)
        params = cam.scale_and_shift(params, scale_xy, shift_xy, copy=True)

        sample = self._geometry(dp, ht, wt, params, edited=region is not None)
        flip = sample.pop("flip")
        if flip:
            image = torch.flip(image, dims=(-1,) if flip == 1 else (-2,))
        return {
            **sample,
            "image": image,
            "original_image_size": np.array((w, h)),
            "scale_xy": scale_xy,
            "shift_xy": shift_xy,
        }


class MergedDataset(Dataset):
    """Several `KnotDataset`s concatenated **in memory**."""

    def __init__(self, datasets: list[KnotDataset], weights: list[float] | None = None,
                 groups: list[int] | None = None):
        assert datasets, "no datasets to merge"
        self.groups = list(range(len(datasets))) if groups is None else list(groups)
        assert len(self.groups) == len(datasets) and sorted(set(self.groups)) == list(
            range(max(self.groups) + 1)), f"bad groups {self.groups} for {len(datasets)} datasets"
        npools = {d.npools for d in datasets}
        assert len(npools) == 1, f"datasets disagree on the geometry sampling pools: {npools}"
        self.datasets = datasets
        self.npools = datasets[0].npools
        self.seed = datasets[0].seed
        reps = [1] * len(datasets) if weights is None else [max(1, round(x)) for x in weights]
        self.index: list[tuple[int, int]] = [
            (d, i) for d, (ds, r) in enumerate(zip(datasets, reps)) for _ in range(r)
            for i in range(len(ds))
        ]
        self.name = "+".join(d.name for d in datasets)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx):
        rands: tuple = ()
        if isinstance(idx, tuple):
            idx, *rands_list = idx
            rands = tuple(rands_list)
        d, i = self.index[idx]
        return self.datasets[d][(i, *rands) if rands else i]


def build_dataset(dataset_dir: Path, split: str, **kwargs) -> KnotDataset:
    """One dataset split, ready for `make_loader` (see `KnotDataset` for the options)."""
    if split == "train" and not kwargs.get("jpeg_bytes"):
        warn_on_augmentation_mismatch(str(kwargs.get("augmentations", "geocalib")))
    return KnotDataset(Path(dataset_dir), split, **kwargs)


def dataset_groups(dataset) -> list[np.ndarray]:
    """Where each source dataset's samples sit in ``dataset``: one index array per sampling share of a
    `MergedDataset` (its members, or the members of one ``groups`` entry together; oversampling repeats
    included), one array of everything for a lone dataset.
    """
    if isinstance(dataset, MergedDataset):
        group = np.asarray(dataset.groups, dtype=np.int64)
        which = group[np.fromiter((d for d, _ in dataset.index), dtype=np.int64, count=len(dataset))]
        return [np.flatnonzero(which == g) for g in range(int(group.max()) + 1)]
    return [np.arange(len(dataset))]


class BatchedRandomSampler:
    """Shuffled indices, each paired with random numbers shared by its whole batch."""

    def __init__(self, dataset, batch_size: int, npools: int, drop_last: bool = False,
                 per_dataset: int | None = None, seed: int = 0):
        self.batch_size = batch_size
        self.seed = seed
        self.npools = npools
        if dist.is_available() and dist.is_initialized():
            self.world_size, self.rank = dist.get_world_size(), dist.get_rank()
        else:
            self.world_size, self.rank = 1, 0
        self.epoch: int | None = None
        self.per_dataset = per_dataset
        self.groups = dataset_groups(dataset) if per_dataset else None
        if per_dataset:
            assert per_dataset > 0 and all(len(g) for g in self.groups), \
                f"per_dataset={per_dataset} needs every dataset to have samples"
        n = per_dataset * len(self.groups) if per_dataset else len(dataset)
        step = batch_size * self.world_size
        self.total_size = (n // step) * step if drop_last else n
        assert self.world_size == 1 or drop_last, "must drop the last batch in distributed mode"

    def __len__(self) -> int:
        return self.total_size // self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _draw_epoch(self, rng) -> np.ndarray:
        """This epoch's indices under ``per_dataset``: that many from each group, shuffled together."""
        parts = []
        for group in self.groups:
            passes = -(-self.per_dataset // len(group))
            order = np.concatenate([rng.permutation(group) for _ in range(passes)])
            parts.append(order[: self.per_dataset])
        idxs = np.concatenate(parts)
        rng.shuffle(idxs)
        return idxs[: self.total_size]

    def __iter__(self):
        if self.epoch is None:
            assert self.world_size == 1, "use set_epoch() in distributed mode"
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777 + 1_000_003 * self.seed
        rng = np.random.default_rng(seed=seed)
        if self.per_dataset:
            idxs = self._draw_epoch(rng)
        else:
            idxs = np.arange(self.total_size)
            rng.shuffle(idxs)
        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size
        rns = np.repeat(rng.random((n_batches, self.npools)), self.batch_size, axis=0)
        rns = rns[: self.total_size]
        per_rank = self.batch_size * (
            (self.total_size + self.world_size * self.batch_size - 1)
            // (self.world_size * self.batch_size)
        )
        idxs = idxs[self.rank * per_rank : (self.rank + 1) * per_rank]
        rns = rns[self.rank * per_rank : (self.rank + 1) * per_rank]
        if self.npools > 0:
            yield from ((int(i), *r) for i, r in zip(idxs, rns))
        else:
            yield from (int(i) for i in idxs)


class ShapeGroupedBatchSampler:
    """Batches of validation samples that share one input shape, with every sample kept."""

    def __init__(self, dataset, batch_size: int):
        groups: dict[tuple, list[int]] = {}
        for i in range(len(dataset)):
            dp, target = dataset._target_size(i)
            groups.setdefault(target or (dp.h, dp.w), []).append(i)
        self.batches = [g[k : k + batch_size] for g in groups.values()
                        for k in range(0, len(g), batch_size)]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def collate(batch):
    """Stack a list of samples: tensors and arrays stacked, strings kept as lists, dicts and sequences recursed
    -- the behaviour of AnyCalib's `collate`.
    """
    elem = batch[0]
    if isinstance(elem, Tensor):
        return torch.stack(batch, 0)
    if isinstance(elem, np.ndarray):
        return torch.stack([torch.as_tensor(b) for b in batch], 0)
    if isinstance(elem, np.generic):
        return torch.as_tensor(np.array(batch))
    if isinstance(elem, bool):
        return torch.tensor(batch)
    if isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    if isinstance(elem, int):
        return torch.tensor(batch)
    if isinstance(elem, str):
        return list(batch)
    if isinstance(elem, dict):
        return {key: collate([d[key] for d in batch]) for key in elem}
    if isinstance(elem, (list, tuple)):
        return [collate(list(samples)) for samples in zip(*batch)]
    if elem is None:
        return None
    return torch.stack(batch, 0)


def worker_init_fn(worker_id: int) -> None:
    """Seed each loader worker from the dataset's seed and its id; one thread per worker."""
    info = get_worker_info()
    seed = int(getattr(info.dataset, "seed", 0)) + info.id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def make_loader(
    dataset: KnotDataset | MergedDataset,
    batch_size: int,
    split: str = "train",
    num_workers: int = 6,
    prefetch_factor: int = 2,
    distributed: bool = False,
    pin_memory: bool = True,
    samples_per_dataset: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """A loader with the batched geometry sampler (one aspect ratio/resolution per batch)."""
    if samples_per_dataset and not (split == "train" and dataset.npools > 0):
        raise ValueError("samples_per_dataset needs the batched training sampler "
                         "(split='train' and at least one geometry pool)")
    base = dataset.datasets[0] if isinstance(dataset, MergedDataset) else dataset
    collate_fn = collate_bytes if getattr(base, "jpeg_bytes", False) else collate
    drop_last = split == "train"
    shuffle = False
    if split == "train" and dataset.npools > 0:
        sampler = BatchedRandomSampler(dataset, batch_size, dataset.npools, drop_last=True,
                                       per_dataset=samples_per_dataset, seed=seed)
    elif distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=drop_last)
    else:
        sampler = None
        shuffle = split == "train"
    batching = dict(batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=drop_last)
    if split != "train" and sampler is None and hasattr(dataset, "_target_size"):
        batching = dict(batch_sampler=ShapeGroupedBatchSampler(dataset, batch_size))
    return DataLoader(
        dataset,
        **batching,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor if num_workers else None,
        persistent_workers=num_workers > 0,
    )


def set_epoch(loader: DataLoader, epoch: int) -> None:
    """Reseed the batched/distributed sampler (both need it, in distributed mode)."""
    sampler = loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def resolve_dataset_dir(name_or_path: str, machine: str | None = None) -> Path:
    """A dataset given as a path, or as a name under one of the view-dataset roots."""
    import paths

    p = Path(name_or_path)
    if p.exists():
        return p
    for key in ("openpano_root", "commons_root", "commons_sa_root",
                "megascenes_root"):
        try:
            candidate = paths.get(key, machine) / name_or_path
        except KeyError:
            continue
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no dataset {name_or_path!r}: not a path, and not found under the dataset roots "
        f"of the '{machine or 'default'}' paths config (see `python paths.py`)"
    )


def summarize(dataset: KnotDataset | MergedDataset) -> str:
    parts = dataset.datasets if isinstance(dataset, MergedDataset) else [dataset]
    counts = ", ".join(f"{len(d)} {d.name}/{d.split}" for d in parts)
    return f"{len(dataset)} samples ({counts})"


def cam_id_counts(dataset: KnotDataset | MergedDataset) -> dict[str, int]:
    """Camera models present, straight from the index (no image is read)."""
    parts = dataset.datasets if isinstance(dataset, MergedDataset) else [dataset]
    counts: dict[str, int] = {}
    for d in parts:
        for dp in d.datapoints:
            counts[str(dp.cam_id)] = counts.get(str(dp.cam_id), 0) + 1
    return dict(sorted(counts.items()))


def collate_bytes(samples: list[dict]) -> dict:
    """`collate` for the bytes path: JPEG buffers have different lengths, so they stay a list."""
    jpegs = [s.pop("jpeg") for s in samples]
    batch = collate(samples)
    batch["jpeg"] = jpegs
    return batch


def quadrature_weights_for(
    batch: dict, k: int, placement: str, device, profile: str = "radial"
) -> Tensor:
    """Per-knot loss weights for the shape of a batch Cached in geometry."""
    if profile == "field":
        n = geometry.field_num_values(k)
        return torch.full((n,), 1.0 / n, device=device)
    if "image" in batch:
        h, w = batch["image"].shape[-2:]
    else:
        h, w = batch["target_size"][0].tolist()
    return geometry.quadrature_weights(int(h), int(w), k, placement, device=device)


def to_device(batch: dict, device, non_blocking: bool = True) -> dict:
    return {
        key: value.to(device, non_blocking=non_blocking) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }

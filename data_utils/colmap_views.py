"""COLMAP reconstructions -> training views: camera parsing, the centered crop and resize, and the per-scene
collection of views (used by `prepare_megascenes.py`)."""

from __future__ import annotations

import collections
import math
import os
import struct
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import influx

ASPECTS = ((1, 1), (3, 4), (2, 3))
COLMAP_MODELS = ("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE",
                 "FULL_OPENCV", "FOV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE")
COLMAP_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL = 0, 1, 2
META_COLUMNS = (
    "fname", "split", "scene", "cam_id", "h", "w", "f", "k1", "h_src", "w_src", "fx_src", "fy_src",
    "crop_h", "crop_w", "crop_frac", "resize_factor", "aspect_dev", "c_offset_px",
    "theta_max_deg", "vfov_deg",
)


def default_workers() -> int:
    """Worker threads for decoding and encoding: the CPUs, at most 16."""
    return max(1, min(16, os.cpu_count() or 1))


Camera = tuple[int, int, int, tuple[float, ...]]


def parse_cameras(blob: bytes) -> dict[int, Camera]:
    """`cameras.bin` -> ``camera id -> (model id, width, height, params)``."""
    (count,) = struct.unpack_from("<Q", blob, 0)
    offset, out = 8, {}
    for _ in range(count):
        camera_id, model, width, height = struct.unpack_from("<iiQQ", blob, offset)
        offset += 24
        n = COLMAP_NUM_PARAMS[model]
        out[camera_id] = (model, width, height, struct.unpack_from(f"<{n}d", blob, offset))
        offset += 8 * n
    assert offset == len(blob), f"cameras.bin: {len(blob) - offset} bytes left over"
    return out


def parse_image_cameras(blob: bytes) -> dict[str, int]:
    """`images.bin` -> ``image name -> camera id``, stepping over the 2D observations."""
    (count,) = struct.unpack_from("<Q", blob, 0)
    offset, out = 8, {}
    for _ in range(count):
        (camera_id,) = struct.unpack_from("<i", blob, offset + 60)
        offset += 64
        end = blob.index(b"\x00", offset)
        name = blob[offset:end].decode("utf-8")
        (num_points,) = struct.unpack_from("<Q", blob, end + 1)
        offset = end + 9 + 24 * num_points
        out[name] = camera_id
    assert offset == len(blob), f"images.bin: {len(blob) - offset} bytes left over"
    return out


def parse_cameras_txt(blob: bytes) -> dict[int, Camera]:
    """`cameras.txt`: one ``CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]`` line per camera."""
    out = {}
    for line in blob.decode("utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        out[int(fields[0])] = (COLMAP_MODELS.index(fields[1]), int(fields[2]), int(fields[3]),
                               tuple(float(v) for v in fields[4:]))
    return out


def parse_image_cameras_txt(blob: bytes) -> dict[str, int]:
    """`images.txt` -> ``image name -> camera id``."""
    lines = blob.split(b"\n")
    first = next((i for i, line in enumerate(lines) if not line.startswith(b"#")), len(lines))
    out = {}
    for line in lines[first::2]:
        fields = line.split(None, 9)
        if len(fields) == 10:
            out[fields[9].decode("utf-8").strip()] = int(fields[8])
    return out


PARSERS = {"cameras.bin": parse_cameras, "images.bin": parse_image_cameras,
           "cameras.txt": parse_cameras_txt, "images.txt": parse_image_cameras_txt}


def store_shape(h: int, w: int, short_side: int) -> tuple[int, int]:
    """The stored size for an ``h x w`` photo: the aspect ratio in `ASPECTS` that keeps the most of it, in the
    photo's own orientation.
    """
    ratio = max(h, w) / min(h, w)
    short, long = max(ASPECTS, key=lambda a: min(ratio / (a[1] / a[0]), (a[1] / a[0]) / ratio))
    assert short_side % short == 0, f"--short_side {short_side} is not a multiple of {short}"
    sides = (short_side, short_side * long // short)
    return sides if w >= h else sides[::-1]


def _with_parity(x: float, like: int, upper: int) -> int:
    """The integer nearest ``x`` that has the parity of ``like`` and does not exceed ``upper``."""
    n = round(x)
    if (n - like) % 2:
        n += 1 if x > n else -1
    while n > upper:
        n -= 2
    return n


def center_crop(h: int, w: int, ht: int, wt: int) -> tuple[int, int, int, int]:
    """``(top, left, crop_h, crop_w)``: the largest crop of aspect ``ht:wt`` that removes the *same* number of
    pixels on both sides, so the image center stays the crop's center.
    """
    if w * ht >= h * wt:
        crop_h, crop_w = h, _with_parity(h * wt / ht, w, w)
    else:
        crop_h, crop_w = _with_parity(w * ht / wt, h, h), w
    return (h - crop_h) // 2, (w - crop_w) // 2, crop_h, crop_w


def build_view(scene: str, name: str, blob: bytes, camera: Camera, args) -> dict | str:
    """One photo -> its store entry (with the encoded JPEG), or the reason it is dropped."""
    import cv2

    model, w, h, params = camera
    k1 = None
    if model == PINHOLE:
        fx, fy, cx, cy = params
    elif model == SIMPLE_PINHOLE:
        fx, cx, cy = params
        fy = fx
    elif model == SIMPLE_RADIAL:
        fx, cx, cy, k1 = params
        fy = fx
    else:
        return "other_model"
    cam_id = "pinhole" if k1 is None else "radial:1"
    if cam_id != args.cam_id:
        return "other_model"
    c_offset = max(abs(cx - 0.5 * w), abs(cy - 0.5 * h))
    if c_offset > args.max_c_offset:
        return "off_center"
    image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        return "undecodable"
    if image.shape[:2] != (h, w):
        return "size_mismatch"

    ht, wt = store_shape(h, w, args.short_side)
    top, left, crop_h, crop_w = center_crop(h, w, ht, wt)
    if min(crop_h, crop_w) < args.min_short_side:
        return "too_small"
    sx, sy = wt / crop_w, ht / crop_h
    aspect_dev = (fy * sy) / (fx * sx) - 1
    if abs(aspect_dev) > args.max_aspect_dev:
        return "not_square"
    f = math.sqrt(fx * sx * fy * sy)
    image = image[top : top + crop_h, left : left + crop_w]
    if (crop_h, crop_w) != (ht, wt):
        image = cv2.resize(image, (wt, ht), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_CUBIC)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    assert ok, f"{scene}/{name}: JPEG encoding failed"
    return {
        "fname": f"{scene}/{name}", "cam_id": cam_id, "h": ht, "w": wt,
        "params": [f, f, 0.5 * wt, 0.5 * ht] + ([] if k1 is None else [k1]),
        "resize_factor": math.sqrt(sx * sy), "scene": scene, "f": f, "k1": 0.0 if k1 is None else k1,
        "h_src": h, "w_src": w, "fx_src": fx, "fy_src": fy,
        "crop_h": crop_h, "crop_w": crop_w, "crop_frac": crop_h * crop_w / (h * w),
        "aspect_dev": aspect_dev, "c_offset_px": c_offset * math.sqrt(sx * sy),
        "jpeg": jpeg.tobytes(),
    }


def finalise_views(records: list[dict], scene: str) -> tuple[list[dict], int]:
    """Check that a scene's views are centered and square, and give each its field of view."""
    import torch

    import gt as gtmod
    from litecalib.anycalib.cameras import CameraFactory

    groups = collections.defaultdict(list)
    for record in records:
        groups[record["cam_id"], record["h"], record["w"]].append(record)
    for (cam_id, h, w), group in groups.items():
        params = torch.tensor([r["params"] for r in group], dtype=torch.float64)
        ok = gtmod.check_center_and_aspect(cam_id, params, h, w)
        assert bool(ok.all()), f"{scene}: {int((~ok).sum())} {h}x{w} views are not centered/square"
        _, theta_max, valid = gtmod.gt_knots(cam_id, params, h, w)
        top = params.new_tensor([0.5 * w, 0.0]).expand(len(group), 1, 2)
        bearing, _ = CameraFactory.create_from_id(cam_id).unproject(params, top)
        vfov = 2 * torch.atan2(torch.linalg.norm(bearing[..., :2], dim=-1), bearing[..., 2])[:, 0]
        for record, tm, fov, keep in zip(group, theta_max.tolist(), vfov.tolist(), valid.tolist()):
            record |= {"valid": keep, "vfov": fov, "theta_max_deg": math.degrees(tm),
                       "vfov_deg": math.degrees(fov)}
    kept = [r for r in records if r["valid"]]
    return kept, len(records) - len(kept)


class Scene:
    """What has gone past of one scene, and the views being made of it."""

    def __init__(self, name: str, args, pool, skip_names: set[str]):
        self.name, self.args, self.pool, self.skip_names = name, args, pool, skip_names
        self.tables: dict[str, dict[str, dict]] = collections.defaultdict(dict)
        self.cameras: dict[str, Camera] | None = None
        self.ignored: set[str] = set()
        self.selected: set[str] | None = None
        self.held: list[tuple[str, bytes]] = []
        self.pending: collections.deque = collections.deque()
        self.records: list[dict] = []
        self.dropped: collections.Counter = collections.Counter()

    def _complete(self) -> list[dict]:
        return [t for t in self.tables.values() if "cameras" in t and "images" in t]

    def feed(self, kind: str, name: str, read: Callable[[], bytes]) -> None:
        if kind != "image":
            self.tables[name][kind.split(".")[0]] = PARSERS[kind](read())
        elif self.cameras is not None or self._close():
            self._take(name, read)
        else:
            self.held.append((name, read()))

    def _close(self) -> bool:
        """Settle which camera each photo has. False if no reconstruction is there yet."""
        tables = sorted(self._complete(), key=lambda t: -len(t["images"]))
        if not tables:
            return False
        self.cameras = {}
        for table in tables:
            small = len(table["images"]) < self.args.min_component
            for photo, camera_id in table["images"].items():
                if photo in self.cameras or photo in self.ignored:
                    continue
                if small:
                    self.ignored.add(photo)
                elif camera_id in table["cameras"]:
                    self.cameras[photo] = table["cameras"][camera_id]
        if self.args.per_scene:
            self.selected = set(influx.even_sample(sorted(self.cameras), self.args.per_scene))
        held, self.held = self.held, []
        for photo, blob in held:
            self._take(photo, lambda b=blob: b)
        return True

    def _take(self, name: str, read: Callable[[], bytes]) -> None:
        if name in self.ignored:
            self.dropped["small_reconstruction"] += 1
        elif name not in self.cameras:
            self.dropped["unregistered"] += 1
        elif self.selected is not None and name not in self.selected:
            pass
        elif name in self.skip_names:
            self.dropped["in_benchmark"] += 1
        else:
            self._submit(build_view, self.name, name, read(), self.cameras[name], self.args)

    def _submit(self, fn, *work) -> None:
        if self.pool is None:
            self._collect(fn(*work))
            return
        self.pending.append(self.pool.submit(fn, *work))
        if len(self.pending) >= 4 * self.args.workers:
            self._collect(self.pending.popleft().result())

    def _collect(self, result: dict | str) -> None:
        if isinstance(result, str):
            self.dropped[result] += 1
        else:
            self.records.append(result)

    def finish(self) -> list[dict] | None:
        """The scene's views in name order, or None if its calibration never turned up."""
        if self.cameras is None and not self._close():
            return None
        while self.pending:
            self._collect(self.pending.popleft().result())
        self.records.sort(key=lambda r: r["fname"])
        self.records, invalid = finalise_views(self.records, self.name)
        if invalid:
            self.dropped["invalid_profile"] += invalid
        return self.records

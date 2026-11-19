"""Puts AnyCalib's training and evaluation code (`siclib`, the `third_party/anycalib` submodule) on `sys.path`.

The inference package vendors the parts of AnyCalib it needs (`litecalib.anycalib`); only the training
augmentations, the panorama view generator and the benchmark download use `siclib`.
"""

import sys
from pathlib import Path

ANYCALIB_ROOT = Path(__file__).resolve().parent / "third_party" / "anycalib"

if not (ANYCALIB_ROOT / "siclib").is_dir():
    raise ImportError(
        f"AnyCalib's code was not found at {ANYCALIB_ROOT}. It is a git submodule: run "
        "`git submodule update --init`."
    )
if str(ANYCALIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYCALIB_ROOT))

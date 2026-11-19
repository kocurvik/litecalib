"""LiteCalib: lightweight single-image camera calibration from a sparse FoV field."""

from pathlib import Path

from litecalib.hub import MODELS
from litecalib.predictor import Calibration, Calibrator

__version__ = "0.1.0"
__all__ = ["Calibration", "Calibrator", "MODELS", "load", "__version__"]


def load(name: str = "litecalib_vitl", device: str | None = None) -> Calibrator:
    """A released model by name (`MODELS`), a Hugging Face repository id (``org/name``) or a local checkpoint
    (a released model's directory or ``.safetensors``, or a training ``.pt``), ready to `calibrate`."""
    if Path(name).expanduser().exists():
        return Calibrator.from_checkpoint(name, device)
    return Calibrator.from_pretrained(name, device)

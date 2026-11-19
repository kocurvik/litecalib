"""The released models on the Hugging Face hub: one repository per model, each holding ``model.safetensors``
and ``config.json``."""

from pathlib import Path

MODELS = {
    "litecalib_vitl": "LiteCalib/litecalib-vitl",
    "litecalib_vitb": "LiteCalib/litecalib-vitb",
    "litecalib_vits": "LiteCalib/litecalib-vits",
    "anycalib_litecalib_data_vitl": "LiteCalib/anycalib-litecalib-data-vitl",
}
WEIGHTS = "model.safetensors"
CONFIG = "config.json"


def repo_id(name: str) -> str:
    """A released model's name (`MODELS`) or a Hugging Face repository id (``org/name``) -> the repository id."""
    if name in MODELS:
        return MODELS[name]
    if name.count("/") == 1 and not Path(name).suffix and not Path(name).exists():
        return name
    raise ValueError(f"unknown model {name!r}: give one of {sorted(MODELS)}, a Hugging Face repository id "
                     "or a local checkpoint")


def download(name: str, revision: str | None = None) -> Path:
    """Download (or find in the local cache) a released model; returns the path of its ``model.safetensors``,
    with ``config.json`` beside it."""
    from huggingface_hub import hf_hub_download

    repo = repo_id(name)
    hf_hub_download(repo, CONFIG, revision=revision)
    return Path(hf_hub_download(repo, WEIGHTS, revision=revision))

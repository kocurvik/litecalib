"""Training checkpoints -> the released models, one folder per model in the layout of its Hugging Face repository:
``model.safetensors`` (weights for inference only; the config also in its metadata), ``config.json`` and the
model card ``README.md``.

    python scripts/export_release.py \
        --ckpt litecalib_vitl=checkpoints/vitl14_..._ms_aux_field_wsd24_k5.pt \
               litecalib_vitb=checkpoints/vitb14_..._ms_aux_field_wsd24_k5.pt \
               litecalib_vits=checkpoints/vits14_..._ms_aux_field_wsd24_k5.pt \
               anycalib_litecalib_data_vitl=checkpoints/vitl14_..._ms_ac_wsd24.pt \
        --out release

prints the upload command of each model, e.g.

    huggingface-cli upload LiteCalib/litecalib-vitl release/litecalib_vitl .
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litecalib.hub import CONFIG, MODELS, WEIGHTS
from litecalib.predictor import build_network, load_checkpoint

GITHUB = "https://github.com/kocurvik/litecalib"
LAVAL = "http://hdrdb.com/indoor-hdr-photometric/"
PAPER = "LiteCalib: Fast, Model-Agnostic Single-View Camera Calibration"
AUTHORS = "Viktor Kocur, Lukáš Bujňák, Lukáš Gajdošech, Torsten Sattler, Zuzana Kukelova"
BACKBONES = {"vits14": "DINOv2 ViT-S/14", "vitb14": "DINOv2 ViT-B/14", "vitl14": "DINOv2 ViT-L/14"}
TRAINING_DATA = f"""## Training data

Trained from DINOv2's weights for 24 epochs (warmup-stable-decay schedule), each epoch 144,000 views: 72,000 drawn
from the panorama views and 72,000 from MegaScenes. All the released models were trained on the same data.

| source | images | views (train / val) | cameras | licences |
|---|---|---|---|---|
| OpenPano, AnyCalib's extended version | 3,651 / 202 panoramas | 51,719 / 2,872 | pinhole, Brown-Conrady (`radial:1`), EUCM, about a third each | per source (below) |
| CommonsPano (`by` and `by-sa` subsets) | 9,336 / 555 panoramas | 148,626 / 8,851 | as OpenPano | public domain, CC0, CC BY (`by`); CC BY-SA (`by-sa`) |
| MegaScenes | 14,382 / 160 scenes | 1,211,520 / 10,779 | `radial:1` from COLMAP's `SIMPLE_RADIAL` | per photo, from Wikimedia Commons (74 % CC BY-SA, 17 % CC BY, 8 % public domain / CC0) |

- **OpenPano**: introduced by GeoCalib, which describes how its panoramas were collected and split
  ([GeoCalib: OpenPano Dataset](https://github.com/cvg/GeoCalib?tab=readme-ov-file#openpano-dataset)); extended by
  AnyCalib with panoramas that need not be aligned with gravity
  ([AnyCalib: Extended OpenPano Dataset](https://github.com/javrtg/AnyCalib?tab=readme-ov-file#extended-openpano-dataset)).
  It consists of tonemapped panoramas from [The Laval Photometric Indoor HDR Dataset]({LAVAL}),
  [PolyHaven](https://polyhaven.com/hdris), [HDRMaps](https://hdrmaps.com/freebies/free-hdris/),
  [AmbientCG](https://ambientcg.com/list?type=hdri&sort=popular) and
  [BlenderKit](https://www.blenderkit.com/asset-gallery?query=category_subtree:hdr). AnyCalib's splits.
- **CommonsPano** ([LiteCalib/CommonsPano](https://huggingface.co/datasets/LiteCalib/CommonsPano)): 360°
  panoramas from Wikimedia Commons, split by uploader, with a per-file attribution list. 465 duplicates were
  removed: 444 of OpenPano panoramas (found by image content and by source name) and 21 within Commons. Three
  training panoramas yield no valid view, so the views come from 9,333 of them.
- **Panorama views**: 640 x 640 perspective and distorted views rendered from each panorama with AnyCalib's camera
  sampling (its `OP_g` recipe), principal point at the center, square pixels.
- **MegaScenes** ([megascenes.github.io](https://megascenes.github.io)): Wikimedia Commons photos with COLMAP
  reconstructions. Each photo is center-cropped to 1:1, 4:3 or 3:2 at a short side of 600 px, its principal point
  at the center. Left out: reconstructions with fewer than 20 photos, failed self-calibrations, and the scenes of the
  MegaDepth evaluation landmarks.
- **Training**: AnyCalib's resolution (about 102,400 pixels, aspect ratio 0.5 to 2) and its photometric
  augmentations.

More in the [repository]({GITHUB}#datasets)."""
BENCHMARKS = ("Stanford2D3D, TartanAir, MegaDepth, LaMAR, MegaDepth-radial, ScanNet++, Mono, Cathedral, InFlux and "
              "InFlux++")
RESULTS = {
    "litecalib_vits": {"ae_median": 1.75, "ae_maa5": 62.2, "vfov_median": 3.75, "vfov_maa5": 39.1,
                       "a100": (5.95, 0.36, "CPU"), "jetson": (10.0, 2.3, "CPU")},
    "litecalib_vitb": {"ae_median": 1.50, "ae_maa5": 66.6, "vfov_median": 3.20, "vfov_maa5": 43.7,
                       "a100": (6.29, 0.32, "CPU"), "jetson": (22.9, 4.2, "CPU")},
    "litecalib_vitl": {"ae_median": 1.15, "ae_maa5": 74.0, "vfov_median": 2.40, "vfov_maa5": 52.3,
                       "a100": (10.14, 0.33, "CPU"), "jetson": (69.8, 9.5, "CPU")},
    "anycalib_litecalib_data_vitl": {"ae_median": 1.14, "ae_maa5": 74.6, "vfov_median": 2.41, "vfov_maa5": 52.4,
                                     "a100": (13.36, 1.69, "GPU"), "jetson": (79.0, 21.6, "GPU")},
}


def results_section(name: str) -> str:
    """Accuracy over the paper's ten benchmarks and runtime on an A100 and a Jetson Orin Nano, if known."""
    r = RESULTS.get(name)
    if r is None:
        return ""
    (an, af, ad), (jn, jf, jd) = r["a100"], r["jetson"]
    return f"""
## Accuracy and runtime

Averaged over the ten benchmarks of the paper ({BENCHMARKS}), each image fitted with its benchmark's camera model:

| median angular error | angular error mAA@5° | median vFoV error | vFoV error mAA@5° |
|---|---|---|---|
| {r['ae_median']:.2f}° | {r['ae_maa5']:.1f} % | {r['vfov_median']:.2f}° | {r['vfov_maa5']:.1f} % |

The angular error is AnyCalib's: the angle between the predicted and the true ray, over the pixels of the image;
mAA@5° is the area under its recall curve up to 5°.

Runtime per image, batch 1 (network + camera fit):

| device | network | fit | total |
|---|---|---|---|
| NVIDIA A100, PyTorch fp16 | {an:.1f} ms | {af:.1f} ms ({ad}) | {an + af:.1f} ms |
| Jetson Orin Nano 8 GB (15 W), TensorRT fp16 | {jn:.1f} ms | {jf:.1f} ms ({jd}) | {jn + jf:.1f} ms |

The fit is the mean over the camera models; the Jetson's network is a TensorRT engine of the exported model,
which this package does not build.
"""


def model_card(name: str, config: dict, n_params: int) -> str:
    """The Hugging Face model card (``README.md``) of a released model."""
    backbone = BACKBONES.get(config["backbone"], config["backbone"])
    if config["arch"] == "anycalib":
        what = (f"[AnyCalib](https://github.com/javrtg/AnyCalib)'s network ({backbone}, dense ray field) trained on "
                "LiteCalib's training data, calibrated with AnyCalib's calibrator on the dense field.")
    else:
        what = (f"LiteCalib with a {backbone} backbone: the camera's field of view at a {config['K']} x {config['K']} "
                "grid of image points, to which any radially symmetric camera model is fitted.")
    return f"""---
license: apache-2.0
library_name: litecalib
tags:
- camera-calibration
- single-view-calibration
- computer-vision
---

# {name}

A released model of **{PAPER}** ({AUTHORS}).

{what} {n_params / 1e6:.0f} M parameters.

Code, documentation and training: [{GITHUB.removeprefix('https://')}]({GITHUB}).

```bash
pip install litecalib
```

```python
import litecalib

model = litecalib.load("{name}")
calib = model.calibrate("photo.jpg", "radial:1")
print(calib.K, calib.distortion)
```
{results_section(name)}
{TRAINING_DATA}

## License

Apache License 2.0. We thank the authors of [The Laval Photometric Indoor HDR Dataset]({LAVAL}) for allowing us to
release these weights under this license.
"""


def export(name: str, ckpt: Path, out_dir: Path) -> Path:
    """Write ``<out_dir>/<name>/`` and check that it loads back into the same network."""
    from safetensors.torch import save_file

    config, state = load_checkpoint(ckpt)
    state = {k: v.detach().contiguous() for k, v in state.items() if not k.startswith("dense.")}
    net = build_network(config, state)
    state = {k: v for k, v in net.state_dict().items()}
    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / WEIGHTS
    save_file(state, str(weights), metadata={"config": json.dumps(config), "source": Path(ckpt).name})
    (folder / CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    n = sum(v.numel() for v in state.values())
    (folder / "README.md").write_text(model_card(name, config, n), encoding="utf-8")
    config2, state2 = load_checkpoint(folder)
    assert config2 == config and state2.keys() == state.keys()
    assert all(torch.equal(state[k], state2[k]) for k in state)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()[:16]
    print(f"{name}: {ckpt} -> {folder} ({n / 1e6:.1f} M parameters, sha256 {digest}), config {config}")
    return folder


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True, metavar="NAME=PATH",
                    help=f"released name (one of {sorted(MODELS)}) and its training checkpoint")
    ap.add_argument("--out", type=Path, default=Path("release"))
    args = ap.parse_args(argv)
    uploads = []
    for item in args.ckpt:
        name, _, path = item.partition("=")
        if name not in MODELS:
            print(f"warning: {name!r} is not in litecalib.hub.MODELS; add it there to make it loadable by name")
        folder = export(name, Path(path), args.out)
        uploads.append(f"huggingface-cli upload {MODELS.get(name, 'LiteCalib/<repo>')} {folder.as_posix()} .")
    print("\nupload:\n" + "\n".join(uploads))


if __name__ == "__main__":
    main()

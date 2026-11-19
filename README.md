# LiteCalib: Fast, Model-Agnostic Single-View Camera Calibration

Viktor Kocur<sup>1</sup>, Lukáš Bujňák<sup>1</sup>, Lukáš Gajdošech<sup>1</sup>, Torsten Sattler<sup>2</sup>,
Zuzana Kukelova<sup>3</sup>

<sup>1</sup> Faculty of Mathematics, Physics and Informatics, Comenius University in Bratislava<br>
<sup>2</sup> Czech Institute of Informatics, Robotics and Cybernetics, Czech Technical University in Prague<br>
<sup>3</sup> Visual Recognition Group, Faculty of Electrical Engineering, Czech Technical University in Prague

**Lightweight single-image camera calibration.** Given one image, LiteCalib estimates the camera's
intrinsics (focal length and lens distortion) for any radially symmetric camera model: pinhole,
Brown-Conrady, Kannala-Brandt, UCM or the division model.

Dense methods such as [AnyCalib](https://github.com/javrtg/AnyCalib) predict a ray for every pixel and
then fit a camera model to all of them. LiteCalib instead predicts the camera's **field of view at a
5 x 5 grid of image points**: for each point, the angles `(atan2(X, Z), atan2(Y, Z))` of its ray. A small
query decoder on a DINOv2 backbone reads these 25 points out of the image tokens; no dense decoder runs at
inference. The camera model is then fitted to the 25 rays with AnyCalib's calibrator, compiled with Numba,
with the principal point at the image center and square pixels.

During training, a dense branch (AnyCalib's DPT decoder and ray head) supervises the backbone with
per-pixel rays. It is dropped at inference, so it costs nothing at test time.

| model | Hugging Face | backbone | params |
|---|---|---|---|
| `litecalib_vitl` | [LiteCalib/litecalib-vitl](https://huggingface.co/LiteCalib/litecalib-vitl) | DINOv2 ViT-L/14 | 338 M |
| `litecalib_vitb` | [LiteCalib/litecalib-vitb](https://huggingface.co/LiteCalib/litecalib-vitb) | DINOv2 ViT-B/14 | 106 M |
| `litecalib_vits` | [LiteCalib/litecalib-vits](https://huggingface.co/LiteCalib/litecalib-vits) | DINOv2 ViT-S/14 | 27 M |
| `anycalib_litecalib_data_vitl` | [LiteCalib/anycalib-litecalib-data-vitl](https://huggingface.co/LiteCalib/anycalib-litecalib-data-vitl) | AnyCalib's network, trained on our data | 320 M |

## Installation

```bash
pip install litecalib
```

This installs the inference package: the networks, the compiled calibrator and the camera models. It needs
PyTorch and Numba. The weights are downloaded from the Hugging Face hub (one repository per model, under
[LiteCalib](https://huggingface.co/LiteCalib)) on first use.

## Usage

```python
import litecalib

model = litecalib.load("litecalib_vitl")         # or litecalib_vitb / litecalib_vits / a local checkpoint

calib = model.calibrate("photo.jpg", cam_id="radial:1")
calib.K            # (1, 3, 3) calibration matrix, in pixels of the original image
calib.distortion   # (1, 1) tensor [[k1]]
undistorted = calib.undistort("photo.jpg")      # (1, 3, H, W) in [0, 1], a pinhole image

# several photos taken with one camera: fit a single camera to all of them
calib = model.calibrate(["a.jpg", "b.jpg", "c.jpg"], cam_id="kb:4", shared=True)
undistorted = calib.undistort(["a.jpg", "b.jpg", "c.jpg"])   # the same camera applied to each
K, dist = calib.to_opencv()                                   # for cv2.fisheye
```

[`notebooks/reconstruction.ipynb`](notebooks/reconstruction.ipynb)
([Colab](https://colab.research.google.com/github/kocurvik/litecalib/blob/main/notebooks/reconstruction.ipynb))
calibrates ten fisheye photos from Zip-NeRF with LiteCalib (the one-parameter division model), undistorts them and
reconstructs the scene with Pi3X, with and without the calibration.

From the command line, printing each calibration as JSON:

```bash
litecalib photo1.jpg photo2.jpg --cam kb:4 --model litecalib_vits --undistort_dir undistorted/
litecalib frames/*.jpg --cam radial:2 --shared        # one camera for all frames
```

## API

### `litecalib.load(name="litecalib_vitl", device=None) -> Calibrator`

Loads a released model by name (`litecalib_vitl`, `litecalib_vitb`, `litecalib_vits`, `anycalib_litecalib_data_vitl`)
or by Hugging Face repository id (`LiteCalib/litecalib-vitl`), downloaded from the hub and cached, or a local
checkpoint: a released model's folder (`model.safetensors` + `config.json`), its `.safetensors` file, or a
training `.pt`.
`device` is where the network runs (default: CUDA if available).

### `Calibrator.calibrate(images, cam_id="pinhole", shared=False, ...) -> Calibration`

Predicts the rays and fits the camera model in one call.

| argument | default | meaning |
|---|---|---|
| `images` | | a path, a PIL image (any mode; converted to RGB), an `(H, W, 3)` uint8 RGB array, a `(3, H, W)` or `(B, 3, H, W)` tensor in `[0, 1]`, or a list of these; all images of one call must have the same size |
| `cam_id` | `"pinhole"` | the camera model to fit (table below) |
| `shared` | `False` | `False`: one camera per image. `True`: one camera fitted jointly to the rays of all images (images taken with one camera at one zoom) |
| `fixed_center` | `True` for LiteCalib, `False` for AnyCalib models | hold the principal point at the image center; `False` estimates it (for cropped images) |
| `square_pixels` | as `fixed_center` | one focal length; `False` fits `fx` and `fy` separately (for stretched images); needs `fixed_center` |
| `iters` | `2` | Gauss-Newton iterations of the calibrator |
| `fit_device` | `"cpu"` | where the fit runs: `"cpu"` (Numba) or `"cuda"` (Numba CUDA) |
| `batch_size` | `8` | images per network forward |

The defaults are the fit used in the paper. With `square_pixels`, `fx` and `fy` can still differ by a fraction
of a pixel: the camera is fitted in the network's resized input and mapped back to the original image, and
the two axes are scaled slightly differently.

Supported camera models, in AnyCalib's notation (the first compiled call of each takes a few seconds while
Numba compiles it; it is cached afterwards):

| `cam_id` | model | distortion coefficients |
|---|---|---|
| `pinhole` | pinhole | none |
| `radial:1` ... `radial:3` | Brown-Conrady, radial terms only (OpenCV's `k1, k2, k3`) | `k1[, k2, k3]` |
| `kb:1` ... `kb:4` | Kannala-Brandt (OpenCV's fisheye model with `kb:4`) | `k1, ..., k4` |
| `ucm` | unified camera model | `xi` |
| `division:1`, `division:2` | division model | `k1[, k2]` |

### `Calibration`

The result: B cameras for B images, or one camera when `shared=True`. All values are in pixels of the
original images, with the origin at the top-left corner of the top-left pixel (AnyCalib's convention).

| member | type | meaning |
|---|---|---|
| `cam_id` | `str` | the fitted model |
| `intrinsics` | `(B, D)` float64 tensor | `fx, fy, cx, cy` followed by the distortion coefficients |
| `K` | `(B, 3, 3)` tensor | calibration matrices |
| `distortion` | `(B, nk)` tensor | distortion coefficients, named by `distortion_names` (e.g. `["k1", "k2"]`) |
| `image_size` | `(h, w)` | size of the calibrated images |
| `num_images`, `shared` | `int`, `bool` | how many images were calibrated, and whether jointly |
| `camera` | AnyCalib camera | the model itself: `camera.project(intrinsics, points)`, `camera.unproject(intrinsics, pixels)` |
| `fov()` | `{"vfov", "hfov"}` | fields of view in degrees, `(B,)` each |
| `undistort(images, scale=1.0, size=None)` | `(B, 3, h, w)` tensor | the images resampled to the pinhole camera `undistorted_K(scale, size)` (black where the camera sees nothing); one camera is applied to every image given, B cameras need B images |
| `undistorted_K(scale=1.0, size=None)` | `(B, 3, 3)` tensor | the intrinsics of those pinhole images: the focal lengths times `scale` (`< 1` zooms out, keeping more of a wide field of view), the principal point kept, or at the center of an output of `size` (h, w) |
| `to_opencv()` | `(K, dist)` numpy | `pinhole` / `radial:K`: `dist = (k1, k2, 0, 0, k3)` for `cv2.undistort`; `kb:4`: the four coefficients of `cv2.fisheye`; the principal point is shifted by -0.5 to OpenCV's pixel-center origin |
| `calib[i]` | `Calibration` | the camera of image `i` |
| `to_dict()`, `Calibration.from_dict(d)` | | plain values, e.g. to save as JSON and load later |

### Lower level

`Calibrator.predict_rays(images, batch_size=8)` runs only the network. It returns `rays`, `(B, N, 3)` unit
rays (N = 25 for LiteCalib, every pixel for AnyCalib models), at the pixels `x`, `(N, 2)`, of the resized
network input `(ht, wt)`, together with the map from original pixels to input pixels (`scale_xy`,
`shift_xy`). `Calibrator.fit(pred, cam_id, shared=False, ...)` fits a camera model to that output and takes the
same options as `calibrate`. To try several camera models on the same images, predict once and fit each:

```python
pred = model.predict_rays(["a.jpg", "b.jpg"])
calibs = {cam: model.fit(pred, cam, shared=True) for cam in ("radial:1", "kb:4", "division:1")}
```

## Repository

The repository also contains everything needed to reproduce the paper: data preparation, training and
evaluation.

```
litecalib/        the pip package: model.py (network), predictor.py (calibration API), calibrator_{cpu,cuda}.py
                  (AnyCalib's calibrator in Numba), geometry.py, fit.py, anycalib/ (vendored from AnyCalib)
train/            train.py (LiteCalib and the ablations), train_anycalib.py (AnyCalib's network), losses
eval/             evaluate.py (benchmarks, efficiency), sparse.py (AnyCalib's calibrator on sparse pixels),
                  edit_intrinsics.py (edited images), metrics.py, download_benchmarks.py
data_utils/       panorama view rendering, MegaScenes and benchmark preparation
configs/          training configs of every model in the paper, view generation configs, data paths
data.py, gt.py    data loading and ground truth; paths.py, slurm.py, vendor.py: plumbing
scripts/          export_release.py (training checkpoint -> a released model's folder: weights, config, model card)
third_party/      AnyCalib as a git submodule (its siclib: augmentations, view sampling, benchmark URLs)
```

```bash
git clone --recursive https://github.com/kocurvik/litecalib
cd litecalib
pip install -e ".[train]"
```

`albumentations` must be 1.4.x: newer versions silently ignore some of AnyCalib's augmentation settings
(`train/train.py` refuses to start in that case).

### Data

Set the data roots of your machine in `configs/paths/local.yaml` (or create another file and select it with
`LITECALIB_PATHS=<name>`); `python paths.py` lists them.

**Benchmarks.** AnyCalib's benchmarks are downloaded by `python eval/download_benchmarks.py --sets table2 table3
edit`; ScanNet++ needs a personal token (`data_utils/prepare_scannetpp.py`). InFlux and InFlux++ are built from
InFlux-Real's validation split by `data_utils/prepare_influx_real.py`, and Cathedral from its SfM
reconstruction by `data_utils/prepare_rd.py`.

**Training data.**

- OpenPano: AnyCalib's extended OpenPano panoramas (see AnyCalib's instructions), in
  `<openpano_root>/panoramas/{train,val,test}`.
- CommonsPano: our 360° panoramas from Wikimedia Commons, on the Hugging Face hub at
  [LiteCalib/CommonsPano](https://huggingface.co/datasets/LiteCalib/CommonsPano), in two subsets: `by` (public
  domain / CC0 / CC BY) and `by-sa` (CC BY-SA), each with `train` and `validation` folders. The view generator
  reads `<root>/panoramas/{train,val}`, `commons_root` for `by` and `commons_sa_root` for `by-sa`; download the
  dataset and copy the images there (only the `.jpg` files: each folder also holds a `metadata.csv`):

  ```bash
  hf download LiteCalib/CommonsPano --repo-type dataset --local-dir commons_download
  for pair in by:/data/commons_pano by-sa:/data/commons_pano_sa; do   # <subset>:<commons_root / commons_sa_root>
      subset=${pair%%:*}; root=${pair#*:}
      mkdir -p "$root/panoramas/train" "$root/panoramas/val"
      cp commons_download/$subset/train/*.jpg "$root/panoramas/train/"
      cp commons_download/$subset/validation/*.jpg "$root/panoramas/val/"
      cp commons_download/attribution.csv "$root/"
  done
  ```

  Keep `attribution.csv` with anything derived from it.
- Views are rendered from the panoramas with AnyCalib's camera sampling: `python data_utils/generate_views.py
  openpano_v2_gen` (likewise `commons_gen` and `commons_sa_gen`, and `openpano_v2` for perspective views). Each
  split is written as one HDF5 store. The released models were trained on all three panorama view sets as one
  share and MegaScenes as the other.
- MegaScenes: real photos with COLMAP's radial cameras, `python data_utils/prepare_megascenes.py` (fetched over
  anonymous HTTP from the MegaScenes S3 bucket, scene by scene).

### Training

Each model of the paper has a config in `configs/`:

| config | model |
|---|---|
| `ms_aux_field_wsd24_k5_{vitl,vitb,vits}_opg.yaml` | LiteCalib (the released models) |
| `ms_ac_wsd24_{vitl,vits}_opg.yaml` | AnyCalib's network on the same data (`train/train_anycalib.py`) |
| `ms_field_wsd24_k5_vitl_opg.yaml` | ablation: no dense branch |
| `ms_aux_field_wsd24_vitl_opg.yaml` | ablation: 7 x 7 grid |
| `ms_aux_field_wsd24_k5_logmap_vitl_opg.yaml` | ablation: AnyCalib's log map instead of the FoV field |
| `ms_auxfov_field_wsd24_k5_vitl_opg.yaml` | ablation: dense branch supervised in the FoV field |
| `ms_aux_wsd24_vitl_opg.yaml` | ablation: radial angle profile |
| `ms_aux_div_wsd24_vitl_opg.yaml` | ablation: direct regression of the division model |
| `opg_*`, `pano_*`, `opgms_*`, `e1_aux_field_k5_vitl_opg.yaml` | training data ablations |
| `e1_aux_field_k5_edit_vitl_opp.yaml` | edited (cropped, stretched) images |

```bash
python train/train.py --config configs/ms_aux_field_wsd24_k5_vitl_opg.yaml
python train/train_anycalib.py --config configs/ms_ac_wsd24_vitl_opg.yaml
```

The ViT-L runs used one GPU with batch 24 for about a day. Every script takes `--submitit` to submit itself
to SLURM with the same arguments; set your account and partitions in `slurm.py` (the defaults there are
examples) and the data roots in `configs/paths/cluster.yaml`. For a quick check of the whole path, add
`--limit_train 64 --limit_val 16 --epochs 1`.

### Evaluation

```bash
# LiteCalib: AnyCalib's calibrator on the 25 rays, c at the image center, one focal (the paper's fit)
python eval/evaluate.py --model checkpoints/<ckpt>.pt --data stanford2d3d tartanair megadepth2k lamar2k \
    megadepth2k-radial scannetpp2k monovo2k influx_val influx_pp_val cathedral --efficiency
# the radial-profile and direct-regression ablations
python eval/evaluate.py --model checkpoints/<radial or division ckpt>.pt --fit it2 --data ...
# AnyCalib's released model or AnyCalib's network trained here, its own calibrator on the dense field
python eval/evaluate.py --model anycalib_gen checkpoints/<anycalib ckpt>.pt --data ...
# the sparse-field study: AnyCalib's calibrator on an N x N grid of pixels, PyTorch or compiled
python eval/evaluate.py --model anycalib_gen --anycalib_sampling dense egrid:5 acnb_egrid:5:it2 \
    --anycalib_c free square --fit_device cpu cuda --data ...
# edited images: focal and principal point errors (c, fx and fy free)
python eval/edit_intrinsics.py --model checkpoints/<edit ckpt>.pt anycalib_edit
```

Each benchmark is fitted with its own camera model (`eval/metrics.py`); the metrics are AnyCalib's.

### Releasing

```bash
python scripts/export_release.py --ckpt litecalib_vitl=checkpoints/<vitl>.pt litecalib_vitb=checkpoints/<vitb>.pt \
    litecalib_vits=checkpoints/<vits>.pt anycalib_litecalib_data_vitl=checkpoints/<anycalib vitl>.pt --out release
huggingface-cli upload LiteCalib/litecalib-vitl release/litecalib_vitl .
huggingface-cli upload LiteCalib/litecalib-vitb release/litecalib_vitb .
huggingface-cli upload LiteCalib/litecalib-vits release/litecalib_vits .
huggingface-cli upload LiteCalib/anycalib-litecalib-data-vitl release/anycalib_litecalib_data_vitl .
```

Each folder is one model's repository: `model.safetensors`, `config.json` and the model card `README.md`
(Apache 2.0, the training data with the Laval link).

The PyPI package is built and uploaded by `.github/workflows/release.yml` when a GitHub release is published
(PyPI trusted publishing); the release tag must match `litecalib.__version__`.

## Datasets

The released models are trained on three sources of images:

- **OpenPano** (extended, `openpano_v2`). OpenPano was introduced by GeoCalib, which describes how its panoramas
  were collected and split ([GeoCalib: OpenPano Dataset](https://github.com/cvg/GeoCalib?tab=readme-ov-file#openpano-dataset)).
  AnyCalib extended it with panoramas that need not be aligned with gravity
  ([AnyCalib: Extended OpenPano Dataset](https://github.com/javrtg/AnyCalib?tab=readme-ov-file#extended-openpano-dataset));
  the extended version consists of tonemapped panoramas from
  [The Laval Photometric Indoor HDR Dataset](http://hdrdb.com/indoor-hdr-photometric/),
  [PolyHaven](https://polyhaven.com/hdris), [HDRMaps](https://hdrmaps.com/freebies/free-hdris/),
  [AmbientCG](https://ambientcg.com/list?type=hdri&sort=popular) and
  [BlenderKit](https://www.blenderkit.com/asset-gallery?query=category_subtree:hdr). We use AnyCalib's splits.
- **CommonsPano**: our 360° panoramas from Wikimedia Commons, at
  [LiteCalib/CommonsPano](https://huggingface.co/datasets/LiteCalib/CommonsPano), in two subsets: `by` (public
  domain / CC0 / CC BY) and `by-sa` (CC BY-SA); a per-file attribution list. The released models use both.
- **MegaScenes** ([megascenes.github.io](https://megascenes.github.io)): real photos from Wikimedia Commons with
  COLMAP reconstructions; each photo keeps its Commons licence.

We thank the authors of [The Laval Photometric Indoor HDR Dataset](http://hdrdb.com/indoor-hdr-photometric/) for
allowing us to release the weights of LiteCalib under the Apache License 2.0, and the authors of all the datasets
above for making their images freely available.

The benchmarks used for evaluation are AnyCalib's (downloaded from their original sources by
`eval/download_benchmarks.py`), plus InFlux / InFlux++ ([InFlux](https://github.com/princeton-vl/InFlux), CC BY 4.0)
and ScanNet++ (its own terms of use).

## License and acknowledgements

LiteCalib is released under the Apache License 2.0. It builds on [AnyCalib](https://github.com/javrtg/AnyCalib)
(Apache 2.0): its camera models, calibrator, DPT decoder and vendored DINOv2 are included in
`litecalib/anycalib/`, and `litecalib/calibrator_{cpu,cuda}.py` port its calibrator to Numba (see `NOTICE`).
The backbones are [DINOv2](https://github.com/facebookresearch/dinov2) (Apache 2.0).

The released weights are also under the Apache License 2.0 (see [Datasets](#datasets) for the permission this
relies on).

## Citation

The paper is under review; a citation will be added here.

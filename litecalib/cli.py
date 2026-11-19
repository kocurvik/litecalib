"""Command line: ``litecalib image.jpg [...] --cam radial:1`` prints the calibration of each image as JSON."""

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Single-image camera calibration with LiteCalib.")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--model", default="litecalib_vitl",
                    help="a released model (litecalib_vitl, litecalib_vitb, litecalib_vits, "
                         "anycalib_litecalib_data_vitl) or a checkpoint path")
    ap.add_argument("--cam", default="pinhole",
                    help="camera model to fit: pinhole, radial:K, kb:K, ucm, division:K")
    ap.add_argument("--shared", action="store_true",
                    help="fit one camera to all images (they must have one size)")
    ap.add_argument("--free_center", action="store_true", help="estimate the principal point and both focals")
    ap.add_argument("--device", default=None, help="network device (default: cuda if available)")
    ap.add_argument("--undistort_dir", type=Path, default=None,
                    help="also write each image undistorted to a pinhole camera here")
    args = ap.parse_args(argv)

    import litecalib

    model = litecalib.load(args.model, args.device)
    fixed = False if args.free_center else None
    if args.shared:
        groups = [(args.images, model.calibrate(args.images, args.cam, shared=True, fixed_center=fixed,
                                                square_pixels=fixed))]
    else:
        groups = [([p], model.calibrate(p, args.cam, fixed_center=fixed, square_pixels=fixed))
                  for p in args.images]
    for paths, calib in groups:
        print(json.dumps({"images": [str(p) for p in paths], **calib.to_dict()}))
        if args.undistort_dir is not None:
            from PIL import Image

            args.undistort_dir.mkdir(parents=True, exist_ok=True)
            for path in paths:
                im = calib.undistort(path)[0]
                arr = (im.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
                Image.fromarray(arr).save(args.undistort_dir / f"{path.stem}_undistorted.png")


if __name__ == "__main__":
    main()

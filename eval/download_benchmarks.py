"""Download AnyCalib's evaluation benchmarks, without running its eval scripts."""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vendor import ANYCALIB_ROOT

import data
import paths

GROUPS = {
    "table2": ["stanford2d3d", "tartanair", "megadepth2k", "lamar2k"],
    "table3": ["megadepth2k-radial", "monovo2k", "scannetpp2k"],
    "edit": ["stanford2d3d_edit", "tartanair_edit", "megadepth2k_edit", "lamar2k_edit"],
}


def present(root: Path, name: str) -> bool:
    """Is this benchmark already extracted?"""
    return data.index_files(root / name) is not None


def urls_from_siclib() -> dict[str, str]:
    """Every benchmark URL siclib knows, read out of its eval scripts."""
    found: dict[str, str] = {}
    for script in sorted((ANYCALIB_ROOT / "siclib" / "eval").glob("*_rays.py")):
        text = script.read_text(encoding="utf-8")
        match = re.search(r'"url":\s*"([^"]+)"', text)
        if not match:
            continue
        name = script.stem[: -len("_rays")].replace("_radial", "-radial")
        found[name] = match.group(1)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["table2"],
                    help=f"benchmark names or groups ({', '.join(GROUPS)})")
    ap.add_argument("--root", type=Path, default=None, help="default: the paths config's `benchmarks`")
    ap.add_argument("--paths", default=None, help="machine in configs/paths (default: auto)")
    ap.add_argument("--list", action="store_true", help="only show what is known and what exists")
    args = ap.parse_args()

    root = args.root or paths.get("benchmarks", args.paths)
    urls = urls_from_siclib()

    if args.list:
        print(f"benchmark root: {root}")
        for name, url in sorted(urls.items()):
            here = present(root, name)
            print(f"  {name:<22} {'present' if here else 'missing':<8} {url}")
        for group, names in GROUPS.items():
            print(f"  [{group}] {' '.join(names)}")
        return

    wanted: list[str] = []
    for spec in args.sets:
        wanted += GROUPS.get(spec, [spec])
    root.mkdir(parents=True, exist_ok=True)

    from siclib.eval.utils import download_and_extract_benchmark

    for name in dict.fromkeys(wanted):
        if present(root, name):
            print(f"{name}: already at {root / name}")
            continue
        url = urls.get(name)
        if url is None:
            print(f"{name}: no URL in siclib (ScanNet++ must be obtained manually) - skipping")
            continue
        print(f"{name}: downloading from {url}", flush=True)
        download_and_extract_benchmark(name, url, root)

    print("\nresult:")
    for name in dict.fromkeys(wanted):
        print(f"  {name:<22} {'ok' if present(root, name) else 'MISSING'}")


if __name__ == "__main__":
    main()

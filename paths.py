"""Per-machine data roots, read from ``configs/paths/<machine>.yaml``."""

import os
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "paths"


def default_machine() -> str:
    """The paths file to use: ``$LITECALIB_PATHS`` (a name in configs/paths or a YAML path), else ``local``."""
    return os.environ.get("LITECALIB_PATHS", "local")


@lru_cache(maxsize=8)
def load(machine: str | None = None) -> dict[str, str]:
    """Parse a paths file. `machine` is a name in configs/paths, or a path to a YAML file."""
    machine = machine or default_machine()
    path = Path(machine) if machine.endswith((".yaml", ".yml")) else CONFIG_DIR / f"{machine}.yaml"
    if not path.exists():
        known = ", ".join(sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"no paths config at {path} (known machines: {known})")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get(key: str, machine: str | None = None) -> Path:
    """Look up one data root. Raises if the key is missing from the machine's file."""
    conf = load(machine)
    if key not in conf:
        raise KeyError(f"'{key}' not in paths config for '{machine or default_machine()}' ({sorted(conf)})")
    return Path(conf[key])


if __name__ == "__main__":
    name = default_machine()
    print(f"machine: {name}")
    for k, v in load(name).items():
        print(f"  {k:<18} {v}{'' if Path(v).exists() else '   (MISSING on this machine)'}")

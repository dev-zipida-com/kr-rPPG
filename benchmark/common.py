"""Shared helpers for the benchmark scripts (paths, environment checks, toolbox imports)."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Iterable, List

REPO = Path(__file__).resolve().parents[1]
TOOLBOX = REPO / "third_party" / "rPPG-Toolbox"
CONFIGS = REPO / "benchmark" / "configs"
MODELS = ("physnet", "physformer", "deepphys", "tscan", "efficientphys")
DISPLAY = {"physnet": "PhysNet", "physformer": "PhysFormer", "deepphys": "DeepPhys", "tscan": "TS-CAN",
           "efficientphys": "EfficientPhys"}


def require_env(names: Iterable[str]) -> None:
    """Fail early with a readable message when a placeholder used by the configs is not set."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        sys.exit("set these environment variables first (see README, 'Environment variables'): "
                 + ", ".join(missing))


def work_dir() -> Path:
    require_env(["KR_RPPG_WORK"])
    return Path(os.environ["KR_RPPG_WORK"])


def import_toolbox() -> None:
    """Make the KR-rPPG adaptation of rPPG-Toolbox importable (it uses top-level package names)."""
    path = str(TOOLBOX)
    if path not in sys.path:
        sys.path.insert(0, path)


def expand_configs(items: List[str]) -> List[Path]:
    """Config files, folders or glob patterns (relative to benchmark/configs/ or absolute) -> YAML files.

    A pattern such as 'preprocess/pure_*.yaml' is expanded here, so it works quoted or from any folder."""
    out: List[Path] = []
    for item in items:
        path = Path(item)
        if not path.exists():
            path = CONFIGS / item
        if path.is_dir():
            out.extend(sorted(path.rglob("*.yaml")))
        elif path.is_file():
            out.append(path)
        elif any(char in item for char in "*?["):
            pattern = item if os.path.isabs(item) else str(CONFIGS / item)
            matches = sorted(Path(p) for p in glob.glob(pattern) if p.endswith(".yaml") and os.path.isfile(p))
            if not matches:
                sys.exit(f"no config matches: {item}")
            out.extend(matches)
        else:
            sys.exit(f"no such config or folder: {item}")
    return out

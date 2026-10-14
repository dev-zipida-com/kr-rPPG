#!/usr/bin/env python3
"""Verify the official rPPG-Toolbox weights against checkpoints/checkpoints_manifest.csv and load them safely.

    python checkpoints/verify_checkpoints.py --upstream $RPPG_TOOLBOX_WEIGHTS           # sha256 of every file found
    python checkpoints/verify_checkpoints.py --upstream $RPPG_TOOLBOX_WEIGHTS --load    # + torch.load(weights_only=True)

`--upstream` is the final_model_release/ folder of ubicomplab/rPPG-Toolbox at commit
b7500b848f84ad7f86e277b4612563b69f4f88f9 (third-party files, not re-hosted here); the PURE / UBFC-rPPG -> KR-rPPG
rows of Table 8 score them. No trained weights are distributed with this repository: KR-rPPG models are trained
by the user with the configs in benchmark/configs/. Files listed in the manifest but absent are reported, not
treated as errors. Exit code 1 on any hash mismatch or load failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict

MANIFEST = Path(__file__).resolve().with_name("checkpoints_manifest.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state_dict(path: Path) -> "OrderedDict":
    """torch.load with weights_only=True (no arbitrary code); strips the DataParallel 'module.' prefix."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{path.name}: expected a state_dict, got {type(state).__name__}")
    return OrderedDict((k[len("module."):] if k.startswith("module.") else k, v) for k, v in state.items())


def resolve(row: Dict[str, str], upstream: Path) -> Path:
    """`upstream:final_model_release/<name>` -> <upstream>/<name>."""
    return upstream / Path(row["file"].split(":", 1)[1]).name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", type=Path, required=True,
                        help="final_model_release/ folder of rPPG-Toolbox at commit b7500b8")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    with open(MANIFEST, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ok = missing = bad = 0
    for row in rows:
        path = resolve(row, args.upstream)
        if not path.exists():
            missing += 1
            continue
        if sha256(path) != row["sha256"]:
            print(f"HASH MISMATCH {row['file']}")
            bad += 1
            continue
        if args.load:
            try:
                state = load_state_dict(path)
                n_values = sum(v.numel() for v in state.values() if hasattr(v, "numel") and v.is_floating_point())
                print(f"ok {row['file']} ({len(state)} tensors, {n_values:,} floating-point values)")
            except Exception as exc:  # report and continue with the other files
                print(f"LOAD FAILED {row['file']}: {exc}")
                bad += 1
                continue
        ok += 1
    print(f"verified {ok}, missing {missing}, failed {bad} (manifest rows: {len(rows)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

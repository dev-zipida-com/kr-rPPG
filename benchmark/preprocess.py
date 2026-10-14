#!/usr/bin/env python3
"""Build the preprocessed (face-cropped, normalised, chunked) caches used by the benchmark configs.

    python benchmark/preprocess.py preprocess/kr_physnet.yaml          # one cache
    python benchmark/preprocess.py preprocess                          # every cache (slow, large)
    python benchmark/preprocess.py 'preprocess/kr_*.yaml'              # quoted patterns are expanded here
    python benchmark/preprocess.py benchmark/configs/preprocess/kr_*.yaml   # or let the shell expand full paths

Each config's TEST.DATA section is instantiated with the matching rPPG-Toolbox loader, which reads the raw
videos and writes <CACHED_PATH>/<EXP_DATA_NAME>/*.npy plus a file list. No model is loaded. The caches hold
face crops of the datasets: keep them inside $KR_RPPG_WORK and never publish them.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

from common import TOOLBOX, expand_configs, import_toolbox

LOADERS = {
    "KoreanrPPG": ("dataset.data_loader.KoreanrPPGLoader", "KoreanrPPGLoader"),
    "PURE": ("dataset.data_loader.PURELoader", "PURELoader"),
    "UBFC-rPPG": ("dataset.data_loader.UBFCrPPGLoader", "UBFCrPPGLoader"),
    "MMPD": ("dataset.data_loader.MMPDLoader", "MMPDLoader"),
    "UBFC-PHYS": ("dataset.data_loader.UBFCPHYSLoader", "UBFCPHYSLoader"),
}


def build_cache(config_path: str) -> None:
    import importlib

    from config import get_config

    cfg = get_config(Namespace(config_file=config_path))
    data = cfg.TEST.DATA
    if not data.DO_PREPROCESS:
        sys.exit(f"{config_path}: TEST.DATA.DO_PREPROCESS is false; use a config from benchmark/configs/preprocess/")
    module, name = LOADERS[data.DATASET]
    loader_cls = getattr(importlib.import_module(module), name)
    print(f"[preprocess] {data.DATASET} -> {data.CACHED_PATH}", flush=True)
    loader = loader_cls(name="test", data_path=data.DATA_PATH, config_data=data, device=cfg.DEVICE)
    print(f"[preprocess] {len(loader)} clips cached", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="config files or folders under benchmark/configs/")
    args = parser.parse_args()
    configs = [str(path.resolve()) for path in expand_configs(args.configs)]
    import_toolbox()
    os.chdir(TOOLBOX)  # the toolbox resolves its Haar cascade relative to its own folder
    for path in configs:
        build_cache(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fetch the one rPPG-Toolbox file that KR-rPPG does not distribute (the PhysFormer loss) and verify it.

    python tools/fetch_physformer_loss.py                     # download, check sha256, install
    python tools/fetch_physformer_loss.py --check             # only check the installed file
    python tools/fetch_physformer_loss.py --from-file PATH    # install a copy you already have, e.g. from a
                                                              # clone of rPPG-Toolbox at the pinned commit
    python tools/fetch_physformer_loss.py --dest FILE         # install to FILE (outside the repository);
                                                              # then export KR_RPPG_PHYSFORMER_LOSS=FILE

`neural_methods/loss/PhysFormerLossComputer.py` of rPPG-Toolbox states that it was "modified based on"
https://github.com/radimspetlik/hr-cnn (GPL-3.0), so KR-rPPG does not redistribute it (THIRD_PARTY_NOTICES.md).
PhysFormer training and evaluation need it. This script downloads the unmodified file from rPPG-Toolbox at the
commit `third_party/rPPG-Toolbox/` is based on (one anonymous HTTPS GET of a public file; nothing is sent),
checks its sha256 and writes it to the location the toolbox reads: `$KR_RPPG_PHYSFORMER_LOSS` if that is set
(absolute path), otherwise `third_party/rPPG-Toolbox/neural_methods/loss/` (git-ignored); `--dest` overrides both.
The KR-rPPG change to that loss (target heart rate limited to 40-179 bpm) is applied by
`neural_methods/loss/PhysFormerLossAdapter.py`, not by editing the file. The fetched file keeps its upstream
terms and must not be committed or redistributed with this repository.

The commit, path, URL and sha256 are read from PhysFormerLossAdapter.py of the target toolbox folder, the same
values the toolbox checks at run time. Fail-closed: nothing is written unless the sha256 matches, a differing
file is not replaced without --force, and a failed write leaves no partial file. Exit codes: 0 file present and
verified, 1 missing / hash mismatch / download or write error, 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TOOLBOX = REPO / "third_party" / "rPPG-Toolbox"
ADAPTER_RELPATH = Path("neural_methods") / "loss" / "PhysFormerLossAdapter.py"
MAX_BYTES = 1 << 20  # the pinned file is about 5 kB


def load_pins(toolbox: Path) -> ModuleType:
    """Import PhysFormerLossAdapter.py of `toolbox` by path (standard library only; torch is not imported)."""
    path = toolbox / ADAPTER_RELPATH
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; --toolbox must be the KR-rPPG third_party/rPPG-Toolbox folder")
    spec = importlib.util.spec_from_file_location("krrppg_physformer_loss_pins", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, timeout: float) -> bytes:
    """Anonymous GET of one public file, capped at MAX_BYTES."""
    if not url.startswith("https://"):
        raise ValueError(f"{url}: only https URLs are fetched")
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https URL)
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"{url}: response larger than {MAX_BYTES} bytes")
    return data


def install(data: bytes, destination: Path, expected: str) -> None:
    """Write `data` next to the destination first, re-check it, then rename it into place (no partial file left)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        partial.write_bytes(data)
        if sha256_hex(partial.read_bytes()) != expected:
            raise OSError(f"{partial}: content differs after writing")
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--toolbox", type=Path, default=DEFAULT_TOOLBOX,
                        help="KR-rPPG rPPG-Toolbox folder whose pins are used (default: %(default)s)")
    parser.add_argument("--dest", type=Path,
                        help="install to this file instead; the toolbox then finds it via KR_RPPG_PHYSFORMER_LOSS")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="only verify the installed file")
    mode.add_argument("--from-file", type=Path, help="install this local copy instead of downloading")
    parser.add_argument("--force", action="store_true", help="replace an installed file whose sha256 differs")
    parser.add_argument("--timeout", type=float, default=60.0, help="download timeout in seconds")
    args = parser.parse_args(argv)

    try:
        pins = load_pins(args.toolbox)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        toolbox_reads = pins.upstream_file()  # $KR_RPPG_PHYSFORMER_LOSS or the default location
    except pins.UpstreamLossMissing as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    destination = args.dest.expanduser().resolve() if args.dest is not None else toolbox_reads
    hint = "" if destination == toolbox_reads else \
        f"\nBefore running PhysFormer: export {pins.LOCATION_ENV}={destination}"
    expected = pins.UPSTREAM_SHA256
    present = sha256_hex(destination.read_bytes()) if destination.is_file() else None

    if present == expected:
        print(f"ok: {destination} (sha256 {expected}, rPPG-Toolbox {pins.UPSTREAM_COMMIT[:7]}){hint}")
        return 0
    if args.check:
        state = "missing" if present is None else f"sha256 {present}, expected {expected}"
        print(f"not ready: {destination}: {state}; run {pins.FETCH_COMMAND}", file=sys.stderr)
        return 1
    if present is not None and not args.force:
        print(f"error: {destination} exists with sha256 {present}, expected {expected}; not replaced "
              f"(use --force to replace it)", file=sys.stderr)
        return 1

    if args.from_file is not None:
        source = str(args.from_file)
        try:
            data = args.from_file.read_bytes()
        except OSError as exc:
            print(f"error: cannot read {source}: {exc}", file=sys.stderr)
            return 1
    else:
        source = pins.UPSTREAM_URL
        print(f"downloading {source}")
        try:
            data = download(source, args.timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"error: download failed: {exc}\nDownload the file by hand from\n    {source}\n(or take it from a "
                  f"clone of {pins.UPSTREAM_REPOSITORY} at commit {pins.UPSTREAM_COMMIT}) and run\n"
                  f"    {pins.FETCH_COMMAND} --from-file <path>", file=sys.stderr)
            return 1

    got = sha256_hex(data)
    if got != expected:
        print(f"error: {source} has sha256 {got}, expected {expected} (rPPG-Toolbox {pins.UPSTREAM_COMMIT}); "
              f"nothing written", file=sys.stderr)
        return 1
    try:
        install(data, destination, expected)
    except OSError as exc:
        print(f"error: cannot write {destination}: {exc}; nothing installed", file=sys.stderr)
        return 1
    print(f"installed {destination} (sha256 {got}). This file is not part of KR-rPPG: keep it out of commits.{hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

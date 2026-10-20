"""KR-rPPG adapter around the PhysFormer loss of rPPG-Toolbox (new file by the KR-rPPG authors, see NOTICE.md).

The upstream module `neural_methods/loss/PhysFormerLossComputer.py` is NOT part of this repository: its header
says it was "modified based on" hr-cnn (GPL-3.0), see THIRD_PARTY_NOTICES.md. Fetch the unmodified file from
rPPG-Toolbox at the commit this folder is based on (download + sha256 check) from the repository root with

    python tools/fetch_physformer_loss.py

which places it next to this module (git-ignored). To keep it outside the repository instead, fetch it with
`--dest /absolute/path/PhysFormerLossComputer.py` and set the environment variable KR_RPPG_PHYSFORMER_LOSS to
that path.

This module contains no code of that file. On first use it reads the file once, checks its sha256 against the
pinned value and runs exactly the bytes it checked as module `neural_methods.loss.PhysFormerLossComputer`; a
missing or different file stops PhysFormer with instructions. It applies the one KR-rPPG change on top of the
upstream function without editing it: the target heart rate is limited to HR_MIN_BPM..HR_MAX_BPM (40-179 bpm,
the bins of the upstream frequency grid) before the upstream DLDL loss is evaluated, so that an out-of-band
label falls on the first or last bin instead of an index outside the grid. The result is numerically identical
to the loss of the KR-rPPG manuscript runs (tests/test_physformer_loss.py).
"""

from __future__ import annotations

import functools
import hashlib
import os
import sys
import types
from pathlib import Path

UPSTREAM_REPOSITORY = "https://github.com/ubicomplab/rPPG-Toolbox"
UPSTREAM_COMMIT = "b7500b848f84ad7f86e277b4612563b69f4f88f9"
UPSTREAM_RELPATH = "neural_methods/loss/PhysFormerLossComputer.py"
UPSTREAM_SHA256 = "e6ce7fa579fad778c8d97b9c614ff8e0a088f467d542457f8141e7175e2aa0be"
UPSTREAM_URL = f"https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/{UPSTREAM_COMMIT}/{UPSTREAM_RELPATH}"
UPSTREAM_MODULE = "neural_methods.loss.PhysFormerLossComputer"
UPSTREAM_FILE = Path(__file__).resolve().with_name("PhysFormerLossComputer.py")  # default location, git-ignored
LOCATION_ENV = "KR_RPPG_PHYSFORMER_LOSS"  # optional: absolute path of the fetched file kept elsewhere
FETCH_COMMAND = "python tools/fetch_physformer_loss.py"
_VERIFIED_MARK = "_kr_rppg_verified_sha256"

HR_MIN_BPM = 40.0
HR_MAX_BPM = 179.0


class UpstreamLossMissing(RuntimeError):
    """The upstream PhysFormer loss file has not been fetched, or it is not the pinned version."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upstream_file() -> Path:
    """Where the upstream file is expected: $KR_RPPG_PHYSFORMER_LOSS if set (absolute path), else UPSTREAM_FILE."""
    value = os.environ.get(LOCATION_ENV, "").strip()
    if not value:
        return UPSTREAM_FILE
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise UpstreamLossMissing(f"{LOCATION_ENV} must be an absolute path to the fetched "
                                  f"PhysFormerLossComputer.py, got {value!r}")
    return path


def _fetch_hint(path: Path) -> str:
    dest = "" if path == UPSTREAM_FILE else f" --dest {path}"
    return f"    {FETCH_COMMAND}{dest}"


def _verified_bytes(path: Path) -> bytes:
    """The content of `path` if its sha256 is the pinned one; UpstreamLossMissing (with instructions) otherwise."""
    if not path.is_file():
        raise UpstreamLossMissing(
            f"PhysFormer needs {UPSTREAM_RELPATH} from rPPG-Toolbox (commit {UPSTREAM_COMMIT[:7]}). KR-rPPG does "
            f"not distribute that file (GPL-3.0 ancestry, see THIRD_PARTY_NOTICES.md). Fetch it once, from the "
            f"repository root, with\n{_fetch_hint(path)}\n(to keep it outside the repository, add --dest "
            f"/absolute/path/PhysFormerLossComputer.py and set {LOCATION_ENV} to that path).\nLooked for: {path}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != UPSTREAM_SHA256:
        raise UpstreamLossMissing(
            f"{path} has sha256 {digest}, expected {UPSTREAM_SHA256} (rPPG-Toolbox commit {UPSTREAM_COMMIT[:7]}). "
            f"KR-rPPG reproduces its results with that exact file only; replace it with\n{_fetch_hint(path)} --force")
    return data


def check_upstream_loss(path: Path | None = None) -> Path:
    """Return the path of the verified upstream loss file; raise UpstreamLossMissing (with instructions) otherwise."""
    path = Path(path) if path is not None else upstream_file()
    _verified_bytes(path)
    return path


@functools.lru_cache(maxsize=None)
def upstream_loss_computer():
    """The upstream TorchLossComputer class. The module is built from exactly the bytes whose sha256 was checked
    (read once, no bytecode cache) and registered as UPSTREAM_MODULE; a module of that name imported by other
    means is refused."""
    path = upstream_file()
    data = _verified_bytes(path)
    module = sys.modules.get(UPSTREAM_MODULE)
    if module is not None:
        if getattr(module, _VERIFIED_MARK, None) != UPSTREAM_SHA256 or Path(module.__file__) != path:
            raise UpstreamLossMissing(f"{UPSTREAM_MODULE} was imported from {getattr(module, '__file__', None)} "
                                      f"without the sha256 check; import it only through PhysFormerLossAdapter")
        return module.TorchLossComputer
    module = types.ModuleType(UPSTREAM_MODULE)
    module.__file__ = str(path)
    module.__package__ = UPSTREAM_MODULE.rpartition(".")[0]
    setattr(module, _VERIFIED_MARK, UPSTREAM_SHA256)
    code = compile(data, str(path), "exec", dont_inherit=True)
    sys.modules[UPSTREAM_MODULE] = module
    try:
        exec(code, module.__dict__)  # noqa: S102 (the pinned upstream file, sha256-verified just above)
    except BaseException:
        sys.modules.pop(UPSTREAM_MODULE, None)
        raise
    return module.TorchLossComputer


def dldl_losses(rppg, hr_bpm, frame_rate, std):
    """KL loss, cross-entropy loss and absolute HR-bin error of the upstream
    `cross_entropy_power_spectrum_DLDL_softmax2`, evaluated with the target `hr_bpm` (0-d tensor, bpm)
    limited to [HR_MIN_BPM, HR_MAX_BPM]."""
    bounded_hr = hr_bpm.clamp(min=HR_MIN_BPM, max=HR_MAX_BPM)
    return upstream_loss_computer().cross_entropy_power_spectrum_DLDL_softmax2(rppg, bounded_hr, frame_rate, std)

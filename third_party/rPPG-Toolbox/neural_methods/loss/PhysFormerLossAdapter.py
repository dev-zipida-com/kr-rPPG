"""KR-rPPG adapter around the PhysFormer loss of rPPG-Toolbox (new file by the KR-rPPG authors, see NOTICE.md).

The upstream module `neural_methods/loss/PhysFormerLossComputer.py` is NOT part of this repository: its header
says it was "modified based on" hr-cnn (GPL-3.0), see THIRD_PARTY_NOTICES.md. Fetch the unmodified file from
rPPG-Toolbox at the commit this folder is based on (download + sha256 check) from the repository root with

    python tools/fetch_physformer_loss.py

which places it next to this module (git-ignored). To keep it outside the repository instead, fetch it with
`--dest /absolute/path/PhysFormerLossComputer.py` and set the environment variable KR_RPPG_PHYSFORMER_LOSS to
that path.

On first use this adapter reads the upstream file once and checks its sha256 against the pinned value; a
missing or different file stops PhysFormer with instructions. Without editing that file, a single runtime
source patch hoists the repeated int(target) scalar read out of the 140-bin Python comprehension. The
original SHA, derived runtime SHA and patch ID are recorded separately. Every Gaussian calculation and
subsequent floating-point operation retains its original order. The target heart rate is also limited to
HR_MIN_BPM..HR_MAX_BPM (40-179 bpm,
the bins of the upstream frequency grid) before the upstream DLDL loss is evaluated, so that an out-of-band
label falls on the first or last bin instead of an index outside the grid. The result is numerically identical
to the loss of the KR-rPPG manuscript runs (tests/test_physformer_loss.py).
"""

from __future__ import annotations

import functools
import hashlib
import json
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
_PATCH_MARK = "_kr_rppg_runtime_patch_id"
_RUNTIME_SHA_MARK = "_kr_rppg_runtime_sha256"
RUNTIME_PATCH_ID = "dldl-target-int-hoist-v1"
_SCALAR_STATEMENT = b"        target_distribution = [normal_sampling(int(target), i, std) for i in range(40, 180)]"
_HOISTED_STATEMENT = (
    b"        _kr_target_int = int(target)\n"
    b"        target_distribution = [normal_sampling(_kr_target_int, i, std) for i in range(40, 180)]"
)

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


def _runtime_source(verified_data: bytes) -> bytes:
    """Replace exactly one statement after verification; preserve every numerical operation."""
    if verified_data.count(_SCALAR_STATEMENT) != 1 or b"_kr_target_int" in verified_data:
        raise UpstreamLossMissing("Pinned PhysFormer loss does not match the single scalar-hoist patch")
    return verified_data.replace(_SCALAR_STATEMENT, _HOISTED_STATEMENT, 1)


def _record_runtime(path: Path, runtime_sha: str) -> None:
    folder = os.environ.get("RPPG_JOB_PROVENANCE_DIR", "")
    if folder:
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        record = {"upstream_file": str(path), "upstream_sha256": UPSTREAM_SHA256,
                  "runtime_source_sha256": runtime_sha, "patch_id": RUNTIME_PATCH_ID,
                  "replaced_statements": 1, "source_file_modified": False}
        (target / "physformer_loss_runtime.json").write_text(json.dumps(record, indent=2) + "\n")


@functools.lru_cache(maxsize=None)
def upstream_loss_computer():
    """Load the SHA-pinned source with the named scalar-hoist runtime patch, without modifying its file.
    The module records both source hashes; a module imported without the same verification and patch is refused.
    """
    path = upstream_file()
    data = _verified_bytes(path)
    runtime_data = _runtime_source(data)
    runtime_sha = hashlib.sha256(runtime_data).hexdigest()
    module = sys.modules.get(UPSTREAM_MODULE)
    if module is not None:
        if (getattr(module, _VERIFIED_MARK, None) != UPSTREAM_SHA256 or Path(module.__file__) != path
                or getattr(module, _PATCH_MARK, None) != RUNTIME_PATCH_ID
                or getattr(module, _RUNTIME_SHA_MARK, None) != runtime_sha):
            raise UpstreamLossMissing(f"{UPSTREAM_MODULE} was imported from {getattr(module, '__file__', None)} "
                                      f"without the same sha256 check and runtime patch; import it only through PhysFormerLossAdapter")
        return module.TorchLossComputer
    module = types.ModuleType(UPSTREAM_MODULE)
    module.__file__ = str(path)
    module.__package__ = UPSTREAM_MODULE.rpartition(".")[0]
    setattr(module, _VERIFIED_MARK, UPSTREAM_SHA256)
    setattr(module, _PATCH_MARK, RUNTIME_PATCH_ID)
    setattr(module, _RUNTIME_SHA_MARK, runtime_sha)
    code = compile(runtime_data, str(path), "exec", dont_inherit=True)
    sys.modules[UPSTREAM_MODULE] = module
    try:
        exec(code, module.__dict__)  # noqa: S102 (SHA-pinned source with the single named runtime patch)
        _record_runtime(path, runtime_sha)
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

"""Binds a private folder to one release folder, so a mistyped or swapped path cannot overwrite another release.

The first build writes the same random token to <private_dir>/release_claim.json and to
<output_dir>/.krrppg_claim (beside KR-rPPG/, never inside the published tree) before any
session output; later builds need matching tokens, so the binding follows renamed folders.
Private folders from before claims existed are accepted only for the release folder whose
files their session records describe. All checks only read; they run before anything is
written and again under the build lock. A metadata-only refresh runs the same checks but
writes no claim (it changes nothing outside KR-rPPG/metadata/).
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from . import constants as C
from .safety import PathPolicy, atomic_write_bytes, file_signature

if TYPE_CHECKING:
    from .builder import BuildConfig

CLAIM_FILE = "release_claim.json"
MARKER_FILE = ".krrppg_claim"
BFDT_SETTINGS = ("block_size", "keep_ratio")


class ReleaseConflictError(RuntimeError):
    """The output or private folder belongs to another release; nothing was written."""


def _read_json(path: Path) -> Optional[Dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _bfdt(settings: Dict) -> Dict:
    return {key: settings.get(key) for key in BFDT_SETTINGS}


def has_built_sessions(release_root: Path) -> bool:
    return release_root.is_dir() and next(release_root.glob(f"p*/T*/{C.RELEASE_VIDEO}"), None) is not None


def _outputs_match(release_root: Path, record: Dict) -> bool:
    """True when the files a session record describes are the ones now in this release folder."""
    outputs = record.get("outputs") or {}
    out_dir = release_root / str(record.get("session", ""))
    for name, meta in outputs.items():
        signature = file_signature(out_dir / name)
        if signature is None or (signature[2], signature[3]) != (meta.get("size"), meta.get("mtime_ns")):
            return False
    return bool(outputs)


def _records(cfg: BuildConfig) -> List[Dict]:
    paths = sorted(cfg.records_dir.glob("*.json")) if cfg.records_dir.is_dir() else []
    return [record for record in map(_read_json, paths) if record]


# --- release folder ------------------------------------------------------------------

def _check_claimed(cfg: BuildConfig, claim: Dict, marker: Optional[Dict], built: bool, wanted: Dict) -> None:
    if _bfdt(claim) != wanted:
        raise ReleaseConflictError(f"{cfg.private_dir} was claimed for {_bfdt(claim)}, not {wanted}; "
                                   "use a new --output-dir and --private-dir for another block size")
    if marker is not None:
        if marker.get("token") != claim.get("token"):
            raise ReleaseConflictError(f"{cfg.output_dir} belongs to another private folder than {cfg.private_dir}")
        return
    # A run can stop between writing the claim and the marker; the folder must then be the claimed path
    # and, if it already holds sessions (adopting an older release), they must be this folder's records.
    if claim.get("release_root") != str(cfg.release_root.resolve()):
        raise ReleaseConflictError(f"{cfg.output_dir} is not the release folder {cfg.private_dir} was claimed for "
                                   f"({claim.get('release_root')})")
    _check_records(cfg, built, wanted)


def _check_unclaimed(cfg: BuildConfig, marker: Optional[Dict], built: bool, wanted: Dict) -> None:
    if marker is not None:
        raise ReleaseConflictError(f"{cfg.output_dir} is claimed by another private folder than {cfg.private_dir}")
    _check_records(cfg, built, wanted)


def _check_records(cfg: BuildConfig, built: bool, wanted: Dict) -> None:
    """Without a matching marker, accept a release folder only on the evidence of the session records."""
    records = _records(cfg)
    release_root = str(cfg.release_root.resolve())
    for record in records:
        found = {**_bfdt(record.get("settings", {})), "release_root": record.get("release_root", release_root)}
        if found != {**wanted, "release_root": release_root}:
            raise ReleaseConflictError(f"{cfg.private_dir} holds sessions built for {found}")
    if records and not any(_outputs_match(cfg.release_root, record) for record in records):
        raise ReleaseConflictError(f"the sessions recorded in {cfg.private_dir} are not the files in "
                                   f"{cfg.release_root}; check --output-dir")
    if built and not records:
        raise ReleaseConflictError(f"{cfg.release_root} already holds sessions that {cfg.private_dir} did not build")


def check_release_folder(cfg: BuildConfig) -> None:
    """Raise ReleaseConflictError unless this private folder may build into this release folder."""
    output = cfg.output_dir.resolve()
    if any(part.name == C.RELEASE_DIRNAME for part in (output, *output.parents)):
        raise ReleaseConflictError(f"--output-dir {output} lies inside a {C.RELEASE_DIRNAME}/ release tree; "
                                   f"give the folder that contains {C.RELEASE_DIRNAME}/")
    wanted = _bfdt(cfg.settings())
    claim = _read_json(cfg.private_dir / CLAIM_FILE)
    marker = _read_json(cfg.output_dir / MARKER_FILE)
    built = has_built_sessions(cfg.release_root)
    if claim is not None:
        _check_claimed(cfg, claim, marker, built, wanted)
    else:
        _check_unclaimed(cfg, marker, built, wanted)


# --- claim -----------------------------------------------------------------------------

def _write_json(policy: PathPolicy, path: Path, data: Dict) -> None:
    atomic_write_bytes(policy, path, json.dumps(data, indent=2).encode(), C.PRIVATE_FILE_MODE)


def claim_release(cfg: BuildConfig, policy: PathPolicy) -> None:
    """Under the build lock: re-check, then write the claim and the marker before any other output."""
    check_release_folder(cfg)
    claim = _read_json(cfg.private_dir / CLAIM_FILE)
    if claim is None:
        claim = {"token": secrets.token_hex(16), "release_root": str(cfg.release_root.resolve()),
                 **_bfdt(cfg.settings())}
        _write_json(policy, cfg.private_dir / CLAIM_FILE, claim)
    if _read_json(cfg.output_dir / MARKER_FILE) is None:
        _write_json(policy, cfg.output_dir / MARKER_FILE, {"token": claim["token"]})

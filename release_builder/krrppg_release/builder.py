"""Builds the release: metadata, per-session CSVs, BFDT FFV1 videos and a build report.

Outputs:
  <output_dir>/KR-rPPG/pNNN/T0x/{6 files}, <output_dir>/KR-rPPG/metadata/{4 files}
  <private_dir>/participant_id_map.csv, session_records/*.json, build_report.json

The source tree is read only. The participant folders a build reads (and the metadata
folder) are snapshotted - every entry's type, size, mtime, ctime and inode - before and
after the run, and the report records whether anything changed.

`refresh_metadata` rewrites only KR-rPPG/metadata/ of an existing release (e.g. the 1.2.0
24-column participants.csv) and proves with the same snapshots that nothing else changed.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from . import __version__
from . import constants as C
from .bfdt import check_release_setting
from .catalog import SessionSpec, index_participant_dirs, read_availability, released_sessions, select_sessions
from .ids import RELEASE_ID_RE, assign_release_ids, load_mapping, load_or_create_mapping, parse_mapping_csv
from .clinical import CrfSources
from .metadata import build_release_metadata, file_has_bytes, render_metadata
from .ownership import CLAIM_FILE, MARKER_FILE, check_release_folder, claim_release, has_built_sessions
from .safety import (
    PathPolicy,
    UnsafePathError,
    atomic_write_bytes,
    build_lock,
    copy_file,
    diff_snapshots,
    ensure_root,
    file_signature,
    is_within,
    make_dirs,
    remove_output_file,
    sha256_file,
    snapshot_tree,
)
from .session_files import load_session_tables, render_tables, validate_tables
from .video import source_properties, transcode_to_release

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildConfig:
    source_root: Path
    output_dir: Path
    private_dir: Path
    metadata_dir: Optional[Path] = None
    timestamps: str = "relative"
    id_order: str = "random"
    seed: Optional[int] = None
    participants: Tuple[str, ...] = ()
    workers: int = 4
    block_size: int = C.BLOCK_SIZE
    keep_ratio: float = C.KEEP_RATIO
    expected_frame_size: Optional[Tuple[int, int]] = C.FRAME_SIZE
    protected_roots: Tuple[str, ...] = C.PROTECTED_ROOTS
    crf_xlsx: Optional[Path] = None  # default: <source_root>/$KRRPPG_CRF_FILENAME
    ild_xlsx: Optional[Path] = None  # default: $KRRPPG_ILD_XLSX (CRF item ILD)
    dry_run: bool = False

    def __post_init__(self):
        for name in ("source_root", "output_dir", "private_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("metadata_dir", "crf_xlsx", "ild_xlsx"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(self, "participants", tuple(self.participants))
        object.__setattr__(self, "protected_roots", tuple(str(p) for p in self.protected_roots))
        if self.timestamps not in C.TIMESTAMP_MODES:
            raise ValueError(f"timestamps must be one of {C.TIMESTAMP_MODES}")
        if self.id_order not in C.ID_ORDERS:
            raise ValueError(f"id_order must be one of {C.ID_ORDERS}")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        check_release_setting(self.block_size, self.keep_ratio, self.expected_frame_size)

    @property
    def release_root(self) -> Path:
        return self.output_dir / C.RELEASE_DIRNAME

    @property
    def metadata_source(self) -> Path:
        return self.metadata_dir or self.source_root / C.METADATA_DIRNAME

    @property
    def crf_sources(self) -> CrfSources:
        return CrfSources.for_source(self.source_root, self.crf_xlsx, self.ild_xlsx)

    @property
    def write_roots(self) -> List[Path]:
        return [self.output_dir, self.private_dir]

    @property
    def mapping_path(self) -> Path:
        return self.private_dir / "participant_id_map.csv"

    @property
    def records_dir(self) -> Path:
        return self.private_dir / "session_records"

    @property
    def report_path(self) -> Path:
        return self.private_dir / "build_report.json"

    @property
    def lock_path(self) -> Path:
        return self.private_dir / "build.lock"

    @property
    def claim_path(self) -> Path:
        """Token binding this private folder to one release folder (see ownership.py)."""
        return self.private_dir / CLAIM_FILE

    @property
    def marker_path(self) -> Path:
        """The same token on the output side, beside KR-rPPG/."""
        return self.output_dir / MARKER_FILE

    def policy(self) -> PathPolicy:
        policy = PathPolicy(self.source_root, self.write_roots, self.protected_roots)
        for source in (self.metadata_source,) + self.crf_sources.files():
            if any(is_within(source, root) for root in policy.write_roots):
                raise UnsafePathError(f"input {source} overlaps an output root")
        return policy

    @property
    def metadata_report_path(self) -> Path:
        return self.private_dir / "metadata_report.json"

    def settings(self) -> Dict:
        """Parameters that change session bytes; a record built with other settings is rebuilt.
        builder_version is the version that last changed session outputs (C.SESSION_OUTPUT_VERSION),
        not the package version: a metadata-only release change must not re-encode every video."""
        return {
            "timestamps": self.timestamps,
            "block_size": self.block_size,
            "keep_ratio": self.keep_ratio,
            "builder_version": C.SESSION_OUTPUT_VERSION,
        }

    def as_dict(self) -> Dict:
        return {f.name: (str(v) if isinstance(v, Path) else v) for f in fields(self) for v in [getattr(self, f.name)]}


# --- one session ------------------------------------------------------------------

def _record_path(cfg: BuildConfig, spec: SessionSpec) -> Path:
    return cfg.records_dir / f"{spec.release_id}_{spec.task}.json"


def _source_signatures(spec: SessionSpec) -> Dict[str, list]:
    return {name: list(file_signature(spec.source_dir / name) or []) for name in C.SOURCE_SESSION_FILES}


def _previous_record(cfg: BuildConfig, spec: SessionSpec) -> Optional[Dict]:
    """The earlier record if it still describes exactly this session's current outputs."""
    try:
        record = json.loads(_record_path(cfg, spec).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        record.get("status") != "built"
        or record.get("settings") != cfg.settings()
        or record.get("participant_key") != spec.participant_key
        or record.get("source_dir") != str(spec.source_dir.resolve())
        or record.get("source_signatures") != _source_signatures(spec)
        or set(record.get("outputs", {})) != set(C.RELEASE_SESSION_FILES)
    ):
        return None
    out_dir = spec.release_dir(cfg.release_root)
    for name, meta in record["outputs"].items():
        signature = file_signature(out_dir / name)
        if signature is None or (signature[2], signature[3]) != (meta["size"], meta["mtime_ns"]):
            return None
    return record


def _clear_partials(policy: PathPolicy, out_dir: Path) -> None:
    for leftover in out_dir.glob(".*.partial*"):
        remove_output_file(policy, leftover)


def build_session(cfg: BuildConfig, spec: SessionSpec) -> Dict:
    """Build one release session; never raises (failures are returned as records)."""
    started = time.monotonic()
    base = {"session": spec.label, "participant_key": spec.participant_key, "task": spec.task}
    try:
        previous = _previous_record(cfg, spec)
        if previous is not None:
            return {**base, "status": "skipped", "frames": previous.get("frames", 0),
                    "warnings": previous.get("warnings", [])}
        policy = cfg.policy()
        src = policy.check_source(spec.source_dir)
        source_signatures = _source_signatures(spec)
        tables = load_session_tables(src)
        warnings = validate_tables(tables)
        rendered = render_tables(tables, cfg.timestamps)
        _width, _height, source_fps = source_properties(src / C.SOURCE_VIDEO)
        if abs(source_fps - C.FPS) > 0.01:
            warnings.append(f"source video reports {source_fps} fps; release is written at {C.FPS} fps")

        out_dir = make_dirs(policy, spec.release_dir(cfg.release_root), C.RELEASE_DIR_MODE)
        _clear_partials(policy, out_dir)
        video = transcode_to_release(
            policy,
            src / C.SOURCE_VIDEO,
            out_dir / C.RELEASE_VIDEO,
            expected_frames=len(tables[C.FRAMES_TIMESTAMP]),
            expected_size=cfg.expected_frame_size,
            block_size=cfg.block_size,
            keep_ratio=cfg.keep_ratio,
        )
        for name, data in rendered.items():
            atomic_write_bytes(policy, out_dir / name, data, C.RELEASE_FILE_MODE)
        copy_file(policy, src / C.FACE_BBOXES, out_dir / C.FACE_BBOXES, C.RELEASE_FILE_MODE)

        if _source_signatures(spec) != source_signatures:
            raise RuntimeError("source files changed while the session was being built")
        outputs = {}
        for name in C.RELEASE_SESSION_FILES:
            stat = (out_dir / name).stat()
            outputs[name] = {"sha256": sha256_file(out_dir / name), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        record = {
            **base,
            "status": "built",
            "settings": cfg.settings(),
            "release_root": str(cfg.release_root.resolve()),
            "source_dir": str(src),
            "source_signatures": source_signatures,
            "frames": video.frames,
            "frames_sha256": video.frames_sha256,
            "warnings": warnings,
            "outputs": outputs,
            "seconds": round(time.monotonic() - started, 2),
        }
        atomic_write_bytes(policy, _record_path(cfg, spec), json.dumps(record, indent=2).encode(), C.PRIVATE_FILE_MODE)
        return record
    except Exception as exc:  # reported per session so one bad session does not stop the build
        return {
            **base,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        }


# --- whole release ------------------------------------------------------------------

def _init_worker() -> None:
    cv2.setNumThreads(1)


def _log_result(result: Dict, done: int, total: int) -> Dict:
    level = logging.ERROR if result["status"] == "failed" else logging.INFO
    detail = result.get("error") or f"{result.get('frames', '-')} frames, {result.get('seconds', 0)} s"
    log.log(level, "[%d/%d] %s %s (%s)", done, total, result["session"], result["status"], detail)
    return result


def _run_sessions(cfg: BuildConfig, sessions: List[SessionSpec]) -> List[Dict]:
    total = len(sessions)
    if cfg.workers == 1 or total <= 1:
        return [_log_result(build_session(cfg, spec), i, total) for i, spec in enumerate(sessions, start=1)]
    results = []
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=get_context("spawn"), initializer=_init_worker) as pool:
        futures = {pool.submit(build_session, cfg, spec): spec for spec in sessions}
        for done, future in enumerate(as_completed(futures), start=1):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # e.g. a worker killed by the OOM killer
                result = {"session": spec.label, "participant_key": spec.participant_key, "task": spec.task,
                          "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            results.append(_log_result(result, done, total))
    return sorted(results, key=lambda r: r["session"])


def _stale_entries(cfg: BuildConfig, released: List[SessionSpec]) -> List[str]:
    """Release-tree entries that are not a released session (left over from older builds)."""
    if not cfg.release_root.is_dir():
        return []
    expected = {(spec.release_id, spec.task) for spec in released}
    participants = {spec.release_id for spec in released}
    stale = []
    for entry in sorted(cfg.release_root.iterdir()):
        if entry.name == C.METADATA_DIRNAME:
            continue
        if not (RELEASE_ID_RE.match(entry.name) and entry.is_dir() and entry.name in participants):
            stale.append(entry.name)
            continue
        stale += [f"{entry.name}/{task.name}" for task in sorted(entry.iterdir()) if (entry.name, task.name) not in expected]
    return stale


def _dry_run_mapping(cfg: BuildConfig, keys) -> Dict[str, str]:
    if cfg.mapping_path.is_file():
        return parse_mapping_csv(cfg.mapping_path.read_text(encoding="utf-8"))
    return assign_release_ids(sorted(keys), order=cfg.id_order, seed=cfg.seed)


def build_release(cfg: BuildConfig) -> Dict:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    policy = cfg.policy()  # refuses overlapping roots before anything is written
    check_release_folder(cfg)  # and another release's folders (also on dry runs)
    availability = read_availability(cfg.metadata_source / C.AVAILABILITY_CSV)
    index = index_participant_dirs(cfg.source_root)

    if cfg.dry_run:
        mapping = _dry_run_mapping(cfg, availability)
        sessions = select_sessions(released_sessions(index, availability, mapping), cfg.participants)
        return {"dry_run": True, "sessions_total": len(sessions), "sessions": [s.label for s in sessions]}

    ensure_root(policy, cfg.output_dir, C.RELEASE_DIR_MODE)
    ensure_root(policy, cfg.private_dir, C.PRIVATE_DIR_MODE)
    with build_lock(policy, cfg.lock_path):
        return _locked_build(cfg, policy, availability, index, started_at)


def _locked_build(cfg: BuildConfig, policy: PathPolicy, availability, index, started_at: str) -> Dict:
    claim_release(cfg, policy)  # re-checked under the lock, written before any other output
    mapping, mapping_created = load_or_create_mapping(
        policy, cfg.mapping_path, availability, order=cfg.id_order, seed=cfg.seed
    )
    released = released_sessions(index, availability, mapping)
    sessions = select_sessions(released, cfg.participants)
    watched_roots = sorted({spec.source_dir.parent for spec in sessions}) + [cfg.metadata_source]
    watched_roots += list(cfg.crf_sources.files())
    before = snapshot_tree(watched_roots)

    make_dirs(policy, cfg.release_root, C.RELEASE_DIR_MODE)
    make_dirs(policy, cfg.records_dir, C.PRIVATE_DIR_MODE)
    metadata = build_release_metadata(
        policy, cfg.metadata_source, cfg.crf_sources, cfg.release_root / C.METADATA_DIRNAME, mapping
    )
    log.info("building %d sessions with %d worker(s) into %s", len(sessions), cfg.workers, cfg.release_root)
    results = _run_sessions(cfg, sessions)
    changes = diff_snapshots(before, snapshot_tree(watched_roots))
    stale = _stale_entries(cfg, released)

    by_status = {status: [r for r in results if r["status"] == status] for status in ("built", "skipped", "failed")}
    report = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "builder_version": __version__,
        "config": cfg.as_dict(),
        "settings": cfg.settings(),
        "release_root": str(cfg.release_root),
        "mapping_created": mapping_created,
        "sessions_total": len(sessions),
        "built": len(by_status["built"]),
        "skipped": len(by_status["skipped"]),
        "failed": len(by_status["failed"]),
        "failures": [{k: r.get(k) for k in ("session", "participant_key", "error")} for r in by_status["failed"]],
        "warnings": [f"{r['session']}: {w}" for r in results for w in r.get("warnings", [])],
        "frames_total": sum(r.get("frames", 0) for r in results if r["status"] != "failed"),
        "stale_entries": stale,
        "metadata_sha256": metadata["release_sha256"],
        "data_quality_rule_counts": metadata["quality_rule_counts"],
        "source_entries_watched": len(before),
        "source_unchanged": not changes,
        "source_changes": changes,
    }
    atomic_write_bytes(policy, cfg.report_path, json.dumps(report, indent=2).encode(), C.PRIVATE_FILE_MODE)
    return report


# --- metadata only ------------------------------------------------------------------

def _outside_metadata(cfg: BuildConfig) -> Dict:
    """Signatures of <output_dir> and every entry below it except KR-rPPG/metadata/ and its files."""
    meta = str(cfg.release_root / C.METADATA_DIRNAME)
    return {path: sig for path, sig in snapshot_tree([cfg.output_dir]).items()
            if path != meta and not path.startswith(meta + os.sep)}


def refresh_metadata(cfg: BuildConfig) -> Dict:
    """Rewrite KR-rPPG/metadata/ of an existing release and nothing else.

    No session is built and no claim or marker is written; metadata files whose bytes already
    match are left as they are. The release must already belong to this private folder
    (ownership.check_release_folder, before and again under the build lock) and its ID mapping
    must exist (a new one is never drawn). The output folder outside metadata/ and every input
    are snapshotted before and after; <private_dir>/metadata_report.json records any change.
    """
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if cfg.participants:
        raise ValueError("a metadata refresh always covers every participant; drop --participants")
    policy = cfg.policy()
    check_release_folder(cfg)
    if not has_built_sessions(cfg.release_root):
        raise ValueError(f"{cfg.release_root} holds no built session; a metadata refresh needs an existing release")
    if not cfg.mapping_path.is_file():
        raise ValueError(f"{cfg.mapping_path} is missing; a metadata refresh never draws a new ID mapping")
    availability = read_availability(cfg.metadata_source / C.AVAILABILITY_CSV)
    if cfg.dry_run:
        mapping = load_mapping(cfg.mapping_path, availability)
        release, counts = render_metadata(cfg.metadata_source, cfg.crf_sources, mapping)
        meta_dir = cfg.release_root / C.METADATA_DIRNAME
        return {"dry_run": True, "metadata_only": True,
                "would_write": sorted(name for name, data in release.items() if not file_has_bytes(meta_dir / name, data)),
                "data_quality_rule_counts": counts}
    with build_lock(policy, cfg.lock_path):
        check_release_folder(cfg)  # again under the lock
        return _locked_refresh(cfg, policy, availability, started_at)


def _locked_refresh(cfg: BuildConfig, policy: PathPolicy, availability, started_at: str) -> Dict:
    mapping = load_mapping(cfg.mapping_path, availability)
    inputs = [cfg.metadata_source, *cfg.crf_sources.files()]
    sources_before, release_before = snapshot_tree(inputs), _outside_metadata(cfg)
    metadata = build_release_metadata(policy, cfg.metadata_source, cfg.crf_sources,
                                      cfg.release_root / C.METADATA_DIRNAME, mapping, only_changed=True)
    release_changes = diff_snapshots(release_before, _outside_metadata(cfg))
    source_changes = diff_snapshots(sources_before, snapshot_tree(inputs))
    report = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "builder_version": __version__,
        "mode": "metadata_only",
        "config": cfg.as_dict(),
        "release_root": str(cfg.release_root),
        "metadata_written": metadata["written"],
        "metadata_unchanged": metadata["unchanged"],
        "metadata_sha256": metadata["release_sha256"],
        "data_quality_rule_counts": metadata["quality_rule_counts"],
        "release_entries_watched": len(release_before),
        "release_unchanged_outside_metadata": not release_changes,
        "release_changes_outside_metadata": release_changes,
        "source_entries_watched": len(sources_before),
        "source_unchanged": not source_changes,
        "source_changes": source_changes,
    }
    atomic_write_bytes(policy, cfg.metadata_report_path, json.dumps(report, indent=2).encode(), C.PRIVATE_FILE_MODE)
    return report

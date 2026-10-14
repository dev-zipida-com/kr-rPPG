"""Read-only checks of a built release directory, optionally against the source.

Checks: Figure 2 layout (only pNNN/T0x folders + metadata/, exactly 6 files per
session, no hidden/partial/symlinked entries, no site or group names), metadata
consistency (contiguous p001..pNNN keys shared by all four CSVs, release flags match
the folders), the 24-column participants.csv schema of manuscript Table 6 (exact columns and
order; site, sex and the 0/1 flags only their allowed values; in the numeric columns only multiples
of the column's step written with its decimals - whole numbers for age and other_comorbidity_count -
within PLAUSIBLE_RANGES, sanity limits for unit or column mix-ups, not data-quality rules: since
1.4.0 the builder blanks no value; bmi equal to released_bmi of the row's height and weight) and
its dictionary (re-derived from participants.csv, so every availability count must match),
per-session tables (row counts, one shared timestamp sequence, relative clock starting
at 0), no leaked identifiers or dates, FFV1 videos whose every frame is block-constant
with the expected frame count, and - with the source and the private mapping -
byte-exact equality of every session file and metadata file with a fresh render of the
source (the participant metadata is re-rendered from the CRF).

Messages never quote participants.csv cells: a value outside the allowed categories, off
the grid, outside the plausible range or a bmi that does not match is reported by column and
count only. The warning about categories held by fewer than C.MIN_CATEGORY_COUNT participants
covers the categorical columns only (site, sex, the 0/1 flags); the numeric columns have no
categories, and their values are not quoted.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2

from . import constants as C
from .catalog import index_participant_dirs
from .ids import RELEASE_ID_RE, parse_mapping_csv
from .clinical import (
    CODINGS,
    PARTICIPANT_CATEGORIES,
    PARTICIPANT_COLUMNS,
    PLAUSIBLE_RANGES,
    CrfSources,
    dictionary_header,
    participants_dictionary,
    released_bmi,
)
from .bfdt import check_release_setting
from .metadata import FOLDS_HEADER, read_csv, render_metadata
from .session_files import (
    SessionDataError,
    frame_sequence_warnings,
    load_session_tables,
    read_table,
    render_tables,
)
from .video import VideoError, VideoSummary, bfdt_frames_sha256, inspect_release_video

OLD_KEY_RE = re.compile(r"(?:incheon|saintvincent|seoul)_p\d{3}")
DATE_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}-\d{2}-\d{2}(?!\d)")
EPOCH_RE = re.compile(r"(?<![\d.])1\d{9}(?:\.\d+)?(?!\d)")
PLAIN_NUMBER = re.compile(r"^\d+(\.\d+)?$")
MAX_RELATIVE_SECONDS = Decimal(86400)
FORBIDDEN_NAMES = set(C.SITES)
GROUP_NAME_RE = re.compile(C.SOURCE_GROUP_PATTERN)  # source group folders
TEXT_SUFFIXES = (".csv", ".txt")


@dataclass
class VerifyReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    participants: int = 0
    sessions: int = 0
    frames: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict:
        return {
            "ok": self.ok,
            "participants": self.participants,
            "sessions": self.sessions,
            "frames": self.frames,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class _Options:
    timestamps: str
    expected_frame_size: Optional[Tuple[int, int]]
    block_size: int
    keep_ratio: float
    check_video: bool


# --- metadata / layout ------------------------------------------------------------

def _metadata_keys(name: str, header: List[str], rows: List[List[str]], report: VerifyReport) -> Optional[List[str]]:
    if C.PARTICIPANT_KEY not in header:
        report.errors.append(f"metadata/{name}: no {C.PARTICIPANT_KEY} column")
        return None
    keys = [row[header.index(C.PARTICIPANT_KEY)] for row in rows]
    bad = [key for key in keys if not RELEASE_ID_RE.match(key)]
    if bad:
        report.errors.append(f"metadata/{name}: {len(bad)} participant keys are not release IDs (pNNN)")
    if len(set(keys)) != len(keys):
        report.errors.append(f"metadata/{name}: duplicate participant keys")
    if keys != sorted(keys):
        report.warnings.append(f"metadata/{name}: rows are not sorted by participant key")
    return keys


def _check_metadata(meta_dir: Path, report: VerifyReport) -> Optional[Dict[str, Set[str]]]:
    """Validate metadata/; return release ID -> released tasks (None if unusable)."""
    if not meta_dir.is_dir():
        report.errors.append("metadata/ is missing")
        return None
    names = sorted(os.listdir(meta_dir))
    if names != sorted(C.METADATA_FILES):
        report.errors.append(f"metadata/ contains {names}, expected {sorted(C.METADATA_FILES)}")
    try:
        tables = {name: read_csv(meta_dir / name) for name in C.METADATA_FILES}
    except Exception as exc:  # csv.Error, OSError, ValueError, UnicodeDecodeError
        report.errors.append(f"metadata unreadable: {exc}")
        return None

    keys = {}
    for name in (C.PARTICIPANTS_CSV, C.AVAILABILITY_CSV, C.FOLDS_CSV):
        keys[name] = _metadata_keys(name, *tables[name], report)
    if any(value is None for value in keys.values()):
        return None
    participants = keys[C.PARTICIPANTS_CSV]
    for name in (C.AVAILABILITY_CSV, C.FOLDS_CSV):
        if set(keys[name]) != set(participants):
            report.errors.append(f"metadata/{name}: participants differ from {C.PARTICIPANTS_CSV}")
    if sorted(participants) != [f"p{i:03d}" for i in range(1, len(participants) + 1)]:
        report.errors.append(f"{C.PARTICIPANTS_CSV}: keys are not a contiguous p001..p{len(participants):03d} range")

    _check_participants(tables[C.PARTICIPANTS_CSV], tables[C.DICTIONARY_CSV], report)

    f_header, f_rows = tables[C.FOLDS_CSV]
    if tuple(f_header) != FOLDS_HEADER or any(not row[1].isdigit() for row in f_rows):
        report.errors.append(f"{C.FOLDS_CSV}: expected columns {list(FOLDS_HEADER)} with integer folds, got {f_header}")
    a_header, a_rows = tables[C.AVAILABILITY_CSV]
    if tuple(a_header) != (C.PARTICIPANT_KEY,) + C.AVAILABILITY_COLUMNS:
        report.errors.append(f"{C.AVAILABILITY_CSV}: unexpected columns {a_header}")
        return None
    released: Dict[str, Set[str]] = {}
    for row in a_rows:
        flags = row[1:]
        if any(flag not in ("0", "1") for flag in flags):
            report.errors.append(f"{C.AVAILABILITY_CSV}: invalid flags for {row[0]}: {flags}")
        released[row[0]] = {task for task, flag in zip(C.TASKS, flags) if flag == "1"}
    report.participants = len(participants)
    return released


def _check_participants(participants, dictionary, report: VerifyReport) -> None:
    """participants.csv: the 24-column schema and vocabulary; participants_dictionary.csv: re-derived from it."""
    header, rows = participants
    if not _check_participant_values(header, rows, report):
        d_header, d_rows = dictionary
        if "variable" not in d_header or [row[d_header.index("variable")] for row in d_rows] != header:
            report.errors.append(f"{C.DICTIONARY_CSV}: variables do not match {C.PARTICIPANTS_CSV} columns")
        return
    _check_dictionary(rows, dictionary, report)
    sites = [row[header.index("site")] for row in sorted(rows)]
    if len(set(sites)) > 1 and sites == sorted(sites):
        report.warnings.append(f"{C.PARTICIPANTS_CSV}: release IDs follow site order (not a random numbering)")
    for column, labels in PARTICIPANT_CATEGORIES.items():  # categorical columns only
        index = header.index(column)
        counts = Counter(row[index] for row in rows)
        small = {label: counts[label] for label in labels if 0 < counts[label] < C.MIN_CATEGORY_COUNT}
        if small:
            report.warnings.append(f"{C.PARTICIPANTS_CSV}: {column} categories with fewer than "
                                   f"{C.MIN_CATEGORY_COUNT} participants: {small}")


def _check_participant_values(header: List[str], rows: List[List[str]], report: VerifyReport) -> bool:
    """Exact columns and order; per categorical column the allowed categories, per numeric column the grid
    and the plausible range, bmi against height and weight (blank = missing). Cells are never quoted."""
    if tuple(header) != PARTICIPANT_COLUMNS:
        unexpected = [column for column in header if column not in PARTICIPANT_COLUMNS]
        missing = [column for column in PARTICIPANT_COLUMNS if column not in header]
        report.errors.append(f"{C.PARTICIPANTS_CSV}: columns are not the 24-column schema in order "
                             f"(unexpected {unexpected}, missing {missing})")
        return False
    for index, column in enumerate(header[1:], start=1):
        cells = [row[index] for row in rows]
        if column not in PARTICIPANT_CATEGORIES:
            _check_numeric_column(column, cells, report)
            continue
        allowed = set(PARTICIPANT_CATEGORIES[column]) | {""}
        bad = sum(1 for cell in cells if cell not in allowed)
        if bad:
            report.errors.append(f"{C.PARTICIPANTS_CSV}: {bad} {column} cells hold a value outside "
                                 f"{list(PARTICIPANT_CATEGORIES[column])} (blank = missing)")
    _check_bmi(header, rows, report)
    return True


def _numeric_cell_pattern(decimals: int) -> re.Pattern:
    """Plain decimal text: no sign, exponent or leading zeros, exactly `decimals` decimals (ASCII digits only)."""
    fraction = rf"\.[0-9]{{{decimals}}}" if decimals else ""
    return re.compile(rf"(?:0|[1-9][0-9]*){fraction}")


def _check_numeric_column(column: str, cells: List[str], report: VerifyReport) -> None:
    """A numeric column: every non-blank cell a multiple of the step with the column's decimals (a whole number
    for an integer column), within PLAUSIBLE_RANGES when the column has a range (bmi: _check_bmi)."""
    coding = CODINGS[column]
    pattern, step = _numeric_cell_pattern(coding.decimals), Decimal(coding.step)
    bounds = PLAUSIBLE_RANGES.get(column)
    off_grid = outside = 0
    for cell in cells:
        if cell == "":
            continue
        if not pattern.fullmatch(cell) or Decimal(cell) % step != 0:
            off_grid += 1
        elif bounds is not None and not Decimal(bounds[0]) <= Decimal(cell) <= Decimal(bounds[1]):
            outside += 1
    if off_grid:
        grid = ("whole numbers written without decimals" if coding.kind == "integer"
                else f"a multiple of {coding.step} written with {coding.decimals} decimals")
        report.errors.append(f"{C.PARTICIPANTS_CSV}: {off_grid} {column} cells are not {grid} (blank = missing)")
    if outside:
        report.errors.append(f"{C.PARTICIPANTS_CSV}: {outside} {column} cells lie outside the plausible range "
                             f"{bounds[0]} to {bounds[1]}")


def _check_bmi(header: List[str], rows: List[List[str]], report: VerifyReport) -> None:
    """bmi = released_bmi(height, weight) of the same row: weight / (height / 100)^2 of the released (rounded)
    numbers, one decimal; blank when either is blank. Rows whose height or weight is not a positive number are
    left to the grid and range checks."""
    height, weight, bmi = (header.index(column) for column in ("height", "weight", "bmi"))
    differing = 0
    for row in rows:
        try:
            expected = released_bmi(row[height], row[weight])
        except ValueError:
            continue
        differing += row[bmi] != expected
    if differing:
        report.errors.append(f"{C.PARTICIPANTS_CSV}: {differing} bmi cells differ from weight / (height / 100)^2 of "
                             f"the row's height and weight cells, one decimal (blank when either is blank)")


def _check_dictionary(rows: List[List[str]], dictionary, report: VerifyReport) -> None:
    d_header, d_rows = dictionary
    expected_header = dictionary_header(len(rows))
    if d_header != expected_header:
        report.errors.append(f"{C.DICTIONARY_CSV}: header {d_header} != {expected_header}")
        return
    expected = participants_dictionary(rows)
    if [row[0] for row in d_rows] != list(PARTICIPANT_COLUMNS):
        report.errors.append(f"{C.DICTIONARY_CSV}: variables do not match {C.PARTICIPANTS_CSV} columns")
        return
    differing = [row[0] for row, wanted in zip(d_rows, expected) if row != wanted]
    if differing:
        report.errors.append(f"{C.DICTIONARY_CSV}: rows for {differing} differ from the builder's dictionary "
                             f"(description, coding or availability count)")


def _check_layout(root: Path, released: Optional[Dict[str, Set[str]]], partial: bool, report: VerifyReport) -> List[Path]:
    session_dirs = []
    for entry in sorted(os.listdir(root)):
        path = root / entry
        if entry == C.METADATA_DIRNAME and path.is_dir():
            continue
        if not (RELEASE_ID_RE.match(entry) and path.is_dir()):
            report.errors.append(f"unexpected entry at release root: {entry}")
            continue
        tasks = sorted(os.listdir(path))
        if not tasks:
            report.errors.append(f"{entry}: empty participant folder")
        if released is not None and entry not in released:
            report.errors.append(f"{entry}: participant folder not listed in metadata")
        for task in tasks:
            if task not in C.TASKS or not (path / task).is_dir():
                report.errors.append(f"{entry}: unexpected entry {task}")
                continue
            if released is not None and entry in released and task not in released[entry]:
                report.errors.append(f"{entry}/{task}: present but flagged as not released")
            session_dirs.append(path / task)
    if released is not None and not partial:
        for pid in sorted(released):
            for task in sorted(released[pid]):
                if not (root / pid / task).is_dir():
                    report.errors.append(f"{pid}/{task}: released in metadata but missing")
    return session_dirs


def _scan_tree(root: Path, timestamps: str, report: VerifyReport) -> None:
    """Names and text contents: no hidden/symlinked entries, site names, old keys or dates."""
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root)
            if name.startswith("."):
                report.errors.append(f"hidden or partial file: {rel}")
            if name in FORBIDDEN_NAMES or GROUP_NAME_RE.match(name):
                report.errors.append(f"site/group name in path: {rel}")
            if path.is_symlink():
                report.errors.append(f"symlink in release: {rel}")
        for name in filenames:
            path = Path(dirpath) / name
            if not name.endswith(TEXT_SUFFIXES) or path.is_symlink():
                continue
            rel = path.relative_to(root)
            text = path.read_text(encoding="utf-8", errors="replace")
            if OLD_KEY_RE.search(text):
                report.errors.append(f"{rel}: contains a site_pNNN participant key")
            if timestamps == "relative":
                if DATE_RE.search(text):
                    report.errors.append(f"{rel}: contains a calendar date")
                if rel.parts[0] != C.METADATA_DIRNAME and EPOCH_RE.search(text):
                    report.errors.append(f"{rel}: contains an epoch-like timestamp")


# --- sessions ------------------------------------------------------------------

def _check_clock(label: str, stamps: List[str], timestamps: str, report: VerifyReport) -> None:
    if not all(PLAIN_NUMBER.match(value) for value in stamps):
        report.errors.append(f"{label}: non-numeric timestamps")
        return
    values = [Decimal(value) for value in stamps]
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        report.errors.append(f"{label}: timestamps decrease")
    if timestamps == "relative":
        if values[0] != 0:
            report.errors.append(f"{label}: relative clock does not start at 0 ({stamps[0]})")
        if values[-1] >= MAX_RELATIVE_SECONDS:
            report.errors.append(f"{label}: timestamps look absolute ({stamps[-1]})")


def _check_tables(label: str, session_dir: Path, opts: _Options, report: VerifyReport) -> Optional[int]:
    try:
        tables = {name: read_table(session_dir / name) for name in C.SESSION_TABLES}
    except Exception as exc:  # csv.Error (NUL bytes, oversized fields), OSError, decode errors
        report.errors.append(f"{label}: {exc}")
        return None
    for name, rows in tables.items():
        width = len(C.TABLE_HEADERS[name])
        if any(len(row) != width for row in rows):
            report.errors.append(f"{label}: {name} has rows with the wrong number of columns")
            return None
    n_frames = len(tables[C.FRAMES_TIMESTAMP])
    if n_frames == 0:
        report.errors.append(f"{label}: {C.FRAMES_TIMESTAMP} is empty")
        return None
    reference = [row[1] for row in tables[C.FRAMES_TIMESTAMP]]
    for name, column in C.TIMESTAMP_COLUMN.items():
        if len(tables[name]) != n_frames:
            report.errors.append(f"{label}: {name} has {len(tables[name])} rows for {n_frames} frames")
        elif [row[column] for row in tables[name]] != reference:
            report.errors.append(f"{label}: {name} timestamps differ from {C.FRAMES_TIMESTAMP}")
    frame_ids = [row[0] for row in tables[C.FRAMES_TIMESTAMP]]
    for name in (C.HR_SYNC, C.SPO2_SYNC):
        if [row[0] for row in tables[name]] != frame_ids:
            report.errors.append(f"{label}: {name} frame column differs from {C.FRAMES_TIMESTAMP}")
    if len(tables[C.FACE_BBOXES]) != n_frames:
        report.warnings.append(f"{label}: {C.FACE_BBOXES} has {len(tables[C.FACE_BBOXES])} rows for {n_frames} frames")
    _check_clock(label, reference, opts.timestamps, report)
    if all(PLAIN_NUMBER.match(value) for value in reference):
        stamps = [Decimal(value) for value in reference]
        report.warnings.extend(f"{label}: {w}" for w in frame_sequence_warnings(frame_ids, stamps))
    return n_frames


def _check_video(label: str, path: Path, n_frames: int, opts: _Options, report: VerifyReport) -> Optional[VideoSummary]:
    try:
        summary = inspect_release_video(path, opts.block_size)
    except (VideoError, ValueError) as exc:
        report.errors.append(f"{label}: {exc}")
        return None
    if summary.frames != n_frames:
        report.errors.append(f"{label}: video has {summary.frames} frames, tables have {n_frames}")
    if opts.expected_frame_size and (summary.width, summary.height) != tuple(opts.expected_frame_size):
        report.errors.append(f"{label}: video is {summary.width}x{summary.height}")
    if abs(summary.fps - C.FPS) > 0.01:
        report.warnings.append(f"{label}: video reports {summary.fps} fps")
    if summary.coarser_block_constant:
        coarser = 2 * opts.block_size
        report.warnings.append(f"{label}: every frame is also block-constant at {coarser} px; "
                               f"was the release built with --block-size {coarser}?")
    return summary


def _cross_check(label: str, session_dir: Path, src_dir: Path, video: Optional[VideoSummary],
                 opts: _Options, report: VerifyReport) -> None:
    try:
        expected = render_tables(load_session_tables(src_dir), opts.timestamps)
        expected[C.FACE_BBOXES] = (src_dir / C.FACE_BBOXES).read_bytes()
    except (OSError, SessionDataError, UnicodeDecodeError) as exc:
        report.errors.append(f"{label}: cannot render source {src_dir}: {exc}")
        return
    for name, data in expected.items():
        if (session_dir / name).read_bytes() != data:
            report.errors.append(f"{label}: {name} differs from source")
    if video is not None:
        try:
            count, digest = bfdt_frames_sha256(src_dir / C.SOURCE_VIDEO, opts.block_size, opts.keep_ratio)
        except (VideoError, ValueError) as exc:
            report.errors.append(f"{label}: cannot read source video: {exc}")
            return
        if (count, digest) != (video.frames, video.frames_sha256):
            report.errors.append(f"{label}: {C.RELEASE_VIDEO} differs from BFDT of the source video")


def _check_session(session_dir: Path, opts: _Options, report: VerifyReport, sources: Optional[Dict[str, Path]]) -> None:
    label = f"{session_dir.parent.name}/{session_dir.name}"
    names = set(os.listdir(session_dir))
    expected = set(C.RELEASE_SESSION_FILES)
    if names != expected:
        report.errors.append(f"{label}: files {sorted(names)} != expected {sorted(expected)}")
        if not expected <= names:
            return
    n_frames = _check_tables(label, session_dir, opts, report)
    if n_frames is None:
        return
    video = _check_video(label, session_dir / C.RELEASE_VIDEO, n_frames, opts, report) if opts.check_video else None
    report.sessions += 1
    report.frames += n_frames
    if sources is not None:
        participant_dir = sources.get(session_dir.parent.name)
        if participant_dir is None:
            report.errors.append(f"{label}: release ID not found in the private mapping")
        else:
            _cross_check(label, session_dir, participant_dir / session_dir.name, video, opts, report)


def verify_release(
    release_root,
    *,
    timestamps: str = "relative",
    expected_frame_size: Optional[Tuple[int, int]] = C.FRAME_SIZE,
    block_size: int = C.BLOCK_SIZE,
    keep_ratio: float = C.KEEP_RATIO,
    partial: bool = False,
    check_video: bool = True,
    source_root=None,
    mapping_path=None,
    metadata_source=None,
    workers: int = 1,
    crf_xlsx=None,
    ild_xlsx=None,
) -> VerifyReport:
    if timestamps not in C.TIMESTAMP_MODES:
        raise ValueError(f"timestamps must be one of {C.TIMESTAMP_MODES}")
    check_release_setting(block_size, keep_ratio, expected_frame_size)
    report = VerifyReport()
    root = Path(release_root)
    if not root.is_dir():
        report.errors.append(f"{root} is not a directory")
        return report
    if root.name != C.RELEASE_DIRNAME:
        report.warnings.append(f"release root is named {root.name!r}, Figure 2 uses {C.RELEASE_DIRNAME!r}")
    opts = _Options(timestamps, expected_frame_size, block_size, keep_ratio, check_video)

    released = _check_metadata(root / C.METADATA_DIRNAME, report)
    session_dirs = _check_layout(root, released, partial, report)
    _scan_tree(root, timestamps, report)

    sources = None
    if source_root is not None:
        if mapping_path is None:
            report.errors.append("the source cross-check needs the private ID mapping (--mapping)")
        else:
            mapping = parse_mapping_csv(Path(mapping_path).read_text(encoding="utf-8"))
            index = index_participant_dirs(source_root)
            sources = {release_id: index[key] for key, release_id in mapping.items() if key in index}
            crf = CrfSources.for_source(source_root, crf_xlsx, ild_xlsx)
            meta_src = Path(metadata_source or Path(source_root) / C.METADATA_DIRNAME)
            _cross_check_metadata(root, meta_src, crf, mapping, report)
    for partial_report in _run_session_checks(session_dirs, opts, sources, workers):
        report.errors.extend(partial_report.errors)
        report.warnings.extend(partial_report.warnings)
        report.sessions += partial_report.sessions
        report.frames += partial_report.frames
    return report


def _verify_one_session(session_dir: Path, opts: _Options, sources: Optional[Dict[str, Path]]) -> VerifyReport:
    report = VerifyReport()
    try:
        _check_session(session_dir, opts, report, sources)
    except Exception as exc:  # report and keep checking the remaining sessions
        report.errors.append(f"{session_dir.parent.name}/{session_dir.name}: verification crashed: "
                             f"{type(exc).__name__}: {exc}")
    return report


def _run_session_checks(session_dirs: List[Path], opts: _Options, sources: Optional[Dict[str, Path]],
                        workers: int) -> List[VerifyReport]:
    """Per-session reports in session order (sessions are independent, so they can run in parallel)."""
    if workers <= 1 or len(session_dirs) <= 1:
        return [_verify_one_session(d, opts, sources) for d in session_dirs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker) as pool:
        return list(pool.map(_verify_one_session, session_dirs, [opts] * len(session_dirs),
                             [sources] * len(session_dirs)))


def _init_worker() -> None:
    cv2.setNumThreads(1)


def _cross_check_metadata(root: Path, metadata_source: Path, crf: CrfSources, mapping: Dict[str, str],
                          report: VerifyReport) -> None:
    """Re-render all four metadata files from the source metadata and the CRF; compare bytes."""
    try:
        release, _counts = render_metadata(metadata_source, crf, mapping)
    except Exception as exc:
        report.errors.append(f"cannot render source metadata {metadata_source}: {exc}")
        return
    for name, data in release.items():
        path = root / C.METADATA_DIRNAME / name
        if not path.is_file() or path.read_bytes() != data:
            report.errors.append(f"metadata/{name} differs from source")

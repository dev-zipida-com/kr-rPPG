"""Reading, validating and rewriting the per-session CSV files.

Release rules:
- frames_timestamp.csv is the canonical clock. The synchronized BVP/HR/SpO2 files
  carry the same timestamps after a float64 round trip; the release writes the
  frames_timestamp.csv value into all four files so they share one exact sequence.
- `relative` mode re-expresses timestamps as seconds since the session's first frame
  (exact decimal arithmetic), so the CSVs carry no recording date or time of day.
- Value columns are copied verbatim as text; nothing is re-parsed as float.
"""

from __future__ import annotations

import csv
import io
import re
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

from . import constants as C

Rows = List[List[str]]
Tables = Dict[str, Rows]

_PLAIN_NUMBER = re.compile(r"^\d+(\.\d+)?$")
_SIGNALS = (C.BVP_SYNC, C.HR_SYNC, C.SPO2_SYNC)


class SessionDataError(ValueError):
    """Source session files are inconsistent; the session must not be released as-is."""


def read_table(path) -> Rows:
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    expected = C.TABLE_HEADERS[path.name]
    if not rows or tuple(rows[0]) != expected:
        found = rows[0] if rows else None
        raise SessionDataError(f"{path.name}: header {found} != expected {list(expected)}")
    return rows[1:]


def load_session_tables(session_dir) -> Tables:
    return {name: read_table(Path(session_dir) / name) for name in C.SESSION_TABLES}


def parse_timestamp(value: str) -> Decimal:
    if not _PLAIN_NUMBER.match(value):
        raise SessionDataError(f"timestamp {value!r} is not a plain non-negative decimal")
    return Decimal(value)


def _check_widths(tables: Tables) -> None:
    for name, rows in tables.items():
        width = len(C.TABLE_HEADERS[name])
        for index, row in enumerate(rows, start=1):
            if len(row) != width:
                raise SessionDataError(f"{name}: data row {index} has {len(row)} columns, expected {width}")


def validate_tables(tables: Tables) -> List[str]:
    """Raise SessionDataError on fatal problems; return non-fatal warnings."""
    _check_widths(tables)
    frames = tables[C.FRAMES_TIMESTAMP]
    n_frames = len(frames)
    if n_frames == 0:
        raise SessionDataError(f"{C.FRAMES_TIMESTAMP} has no data rows")
    for name in _SIGNALS:
        if len(tables[name]) != n_frames:
            raise SessionDataError(f"{name} has {len(tables[name])} rows but {C.FRAMES_TIMESTAMP} has {n_frames}")

    reference = [parse_timestamp(row[1]) for row in frames]
    tolerance = Decimal(C.TIMESTAMP_TOLERANCE_S)
    for name in _SIGNALS:
        column = C.TIMESTAMP_COLUMN[name]
        for index, (row, expected) in enumerate(zip(tables[name], reference), start=1):
            if abs(parse_timestamp(row[column]) - expected) > tolerance:
                raise SessionDataError(
                    f"{name}: data row {index} timestamp {row[column]} differs from "
                    f"{C.FRAMES_TIMESTAMP} ({frames[index - 1][1]}) by more than {tolerance} s"
                )
    frame_ids = [row[0] for row in frames]
    for name in (C.HR_SYNC, C.SPO2_SYNC):
        if [row[0] for row in tables[name]] != frame_ids:
            raise SessionDataError(f"{name}: frame column differs from {C.FRAMES_TIMESTAMP}")
    if any(later < earlier for earlier, later in zip(reference, reference[1:])):
        raise SessionDataError(f"{C.FRAMES_TIMESTAMP}: timestamps decrease")

    warnings = []
    n_boxes = len(tables[C.FACE_BBOXES])
    if n_boxes != n_frames:
        warnings.append(f"{C.FACE_BBOXES} has {n_boxes} rows for {n_frames} frames (copied unchanged)")
    warnings.extend(frame_sequence_warnings(frame_ids, reference))
    return warnings


def frame_sequence_warnings(frame_ids: List[str], stamps: List[Decimal]) -> List[str]:
    """Non-fatal oddities of the capture sequence that users of the release should know about."""
    warnings = []
    try:
        numbers = [int(value) for value in frame_ids]
    except ValueError:
        return [f"{C.FRAMES_TIMESTAMP}: frame column is not integer"]
    resets = sum(1 for earlier, later in zip(numbers, numbers[1:]) if later <= earlier)
    if resets:
        duplicates = len(numbers) - len(set(numbers))
        warnings.append(
            f"{C.FRAMES_TIMESTAMP}: frame column goes backwards {resets} time(s) "
            f"({duplicates} duplicate frame IDs; copied unchanged)"
        )
    threshold = Decimal(C.FRAME_GAP_WARN_S)
    gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:]) if later - earlier > threshold]
    if gaps:
        warnings.append(f"{C.FRAMES_TIMESTAMP}: {len(gaps)} inter-frame gap(s) > {threshold} s (max {max(gaps)} s)")
    return warnings


def relative_timestamps(values: List[str]) -> List[str]:
    """Seconds since the first value, computed exactly in decimal."""
    parsed = [parse_timestamp(value) for value in values]
    origin = parsed[0]
    return [format(value - origin, "f") for value in parsed]


def to_csv_bytes(header, rows: Rows) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def render_tables(tables: Tables, mode: str) -> Dict[str, bytes]:
    """Release bytes for the four timestamped CSVs (face_bboxes.csv is copied as-is)."""
    if mode not in C.TIMESTAMP_MODES:
        raise ValueError(f"timestamp mode must be one of {C.TIMESTAMP_MODES}, got {mode!r}")
    source = [row[1] for row in tables[C.FRAMES_TIMESTAMP]]
    stamps = relative_timestamps(source) if mode == "relative" else list(source)
    rendered = {}
    for name, column in C.TIMESTAMP_COLUMN.items():
        rows = tables[name]
        if len(rows) != len(stamps):
            raise SessionDataError(f"{name} has {len(rows)} rows for {len(stamps)} frames")
        out_rows = []
        for row, stamp in zip(rows, stamps):
            new_row = list(row)
            new_row[column] = stamp
            out_rows.append(new_row)
        rendered[name] = to_csv_bytes(C.TABLE_HEADERS[name], out_rows)
    return rendered

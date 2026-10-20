"""Participant metadata: the four Figure 2 release CSVs in KR-rPPG/metadata/.

  participants.csv              the 24 manuscript Table 6 variables, de-identified (clinical.PARTICIPANT_COLUMNS)
  participants_dictionary.csv   one row per participants.csv column
  task_session_availability.csv re-keyed copy of the source file
  fold_assignment.csv           re-keyed copy of the source file

Only the participant key changes in the two copied files; every other cell is kept as text.
Rows are sorted by release ID so the order says nothing about site or recruitment order.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from . import constants as C
from .clinical import PARTICIPANT_COLUMNS, CrfSources, ParticipantTable, build_participant_table, dictionary_header
from .safety import PathPolicy, atomic_write_bytes, make_dirs
from .session_files import to_csv_bytes

Table = Tuple[List[str], List[List[str]]]
FOLDS_HEADER = (C.PARTICIPANT_KEY, "fold")


def read_csv(path) -> Table:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [row for row in csv.reader(fh) if row]
    if not rows:
        raise ValueError(f"{path} is empty")
    header, body = rows[0], rows[1:]
    for index, row in enumerate(body, start=1):
        if len(row) != len(header):
            raise ValueError(f"{path}: data row {index} has {len(row)} columns, expected {len(header)}")
    return header, body


def rekey_rows(header: List[str], rows: List[List[str]], mapping: Dict[str, str], name: str) -> List[List[str]]:
    if C.PARTICIPANT_KEY not in header:
        raise ValueError(f"{name} has no {C.PARTICIPANT_KEY} column")
    column = header.index(C.PARTICIPANT_KEY)
    keys = [row[column] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} has duplicate participant keys")
    if set(keys) != set(mapping):
        missing = sorted(set(mapping) - set(keys))[:5]
        unmapped = sorted(set(keys) - set(mapping))[:5]
        raise ValueError(f"{name}: participants differ from the ID mapping (missing {missing}, unmapped {unmapped})")
    rekeyed = []
    for row in rows:
        new_row = list(row)
        new_row[column] = mapping[row[column]]
        rekeyed.append(new_row)
    return sorted(rekeyed, key=lambda row: row[column])


def render_metadata(source_dir, crf: CrfSources, mapping: Dict[str, str]) -> Tuple[Dict[str, bytes], Dict[str, int]]:
    """(release metadata files, data-quality rule counts); pure apart from reads."""
    source_dir = Path(source_dir)
    table: ParticipantTable = build_participant_table(crf, mapping)
    release = {
        C.PARTICIPANTS_CSV: to_csv_bytes(PARTICIPANT_COLUMNS, table.rows),
        C.DICTIONARY_CSV: to_csv_bytes(dictionary_header(len(table.rows)), table.dictionary),
    }
    for name in (C.AVAILABILITY_CSV, C.FOLDS_CSV):
        header, rows = read_csv(source_dir / name)
        if name == C.FOLDS_CSV and tuple(header) != FOLDS_HEADER:
            raise ValueError(f"{name}: columns {header}, expected {list(FOLDS_HEADER)} (it must not gain columns)")
        release[name] = to_csv_bytes(header, rekey_rows(header, rows, mapping, name))
    return release, table.quality_counts


def file_has_bytes(path: Path, data: bytes) -> bool:
    """True when `path` is a regular file (not a symlink) holding exactly `data`."""
    return path.is_file() and not path.is_symlink() and path.read_bytes() == data


def build_release_metadata(policy: PathPolicy, source_dir, crf: CrfSources, dest_dir, mapping: Dict[str, str],
                           only_changed: bool = False) -> Dict[str, object]:
    """Write the four release files. With `only_changed`, files whose bytes already match are not
    rewritten (they keep their inode and mtime)."""
    release, counts = render_metadata(source_dir, crf, mapping)
    dest_dir = Path(dest_dir)
    make_dirs(policy, dest_dir, C.RELEASE_DIR_MODE)
    written, kept = [], []
    for name, data in release.items():
        if only_changed and file_has_bytes(dest_dir / name, data):
            kept.append(name)
            continue
        atomic_write_bytes(policy, dest_dir / name, data, C.RELEASE_FILE_MODE)
        written.append(name)
    return {
        "release_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in release.items()},
        "written": sorted(written),
        "unchanged": sorted(kept),
        "quality_rule_counts": counts,
    }

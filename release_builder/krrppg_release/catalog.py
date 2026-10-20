"""Which sessions are released (metadata flags) and where their source files live."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from . import constants as C

_PID_DIR = re.compile(r"^p\d{3}$")
_GROUP_DIR = re.compile(C.SOURCE_GROUP_PATTERN)


@dataclass(frozen=True)
class SessionSpec:
    participant_key: str  # site_pNNN (private)
    release_id: str  # pNNN (released)
    task: str
    source_dir: Path

    @property
    def label(self) -> str:
        return f"{self.release_id}/{self.task}"

    def release_dir(self, release_root) -> Path:
        return Path(release_root) / self.release_id / self.task


def index_participant_dirs(source_root) -> Dict[str, Path]:
    """`site_pNNN` -> <source>/<site>/<group>/pNNN; a key found in two groups is an error."""
    index: Dict[str, Path] = {}
    for site in C.SITES:
        site_dir = Path(source_root) / site
        if not site_dir.is_dir():
            continue
        for group_dir in sorted(site_dir.iterdir()):
            if not (group_dir.is_dir() and _GROUP_DIR.match(group_dir.name)):
                continue
            for child in sorted(group_dir.iterdir()):
                if not (child.is_dir() and _PID_DIR.match(child.name)):
                    continue
                key = f"{site}_{child.name}"
                if key in index:
                    raise ValueError(f"participant {key} appears in both {index[key]} and {child}")
                index[key] = child
    return index


def read_availability(path) -> Dict[str, Tuple[str, ...]]:
    """participant_key -> released tasks, from task_session_availability.csv."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        expected = (C.PARTICIPANT_KEY,) + C.AVAILABILITY_COLUMNS
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path}: columns {reader.fieldnames}, expected {list(expected)}")
        rows = list(reader)
    released: Dict[str, Tuple[str, ...]] = {}
    for row in rows:
        key = row[C.PARTICIPANT_KEY]
        if key in released:
            raise ValueError(f"{path}: duplicate participant {key}")
        flags = [row[column] for column in C.AVAILABILITY_COLUMNS]
        if any(flag not in ("0", "1") for flag in flags):
            raise ValueError(f"{path}: invalid release flags for {key}: {flags}")
        released[key] = tuple(task for task, flag in zip(C.TASKS, flags) if flag == "1")
    if not released:
        raise ValueError(f"{path} lists no participants")
    return released


def released_sessions(
    index: Dict[str, Path], availability: Dict[str, Tuple[str, ...]], mapping: Dict[str, str]
) -> List[SessionSpec]:
    sessions = []
    for key, tasks in availability.items():
        if key not in index:
            raise ValueError(f"no source directory for released participant {key}")
        if key not in mapping:
            raise ValueError(f"released participant {key} has no release ID")
        sessions.extend(SessionSpec(key, mapping[key], task, index[key] / task) for task in tasks)
    return sorted(sessions, key=lambda spec: (spec.release_id, spec.task))


def select_sessions(sessions: List[SessionSpec], participants: Iterable[str]) -> List[SessionSpec]:
    """Restrict to participants given as `site_pNNN` keys or `pNNN` release IDs (empty = all)."""
    wanted = set(participants)
    if not wanted:
        return list(sessions)
    known = {spec.participant_key for spec in sessions} | {spec.release_id for spec in sessions}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"unknown or unreleased participants: {unknown}")
    return [spec for spec in sessions if spec.participant_key in wanted or spec.release_id in wanted]

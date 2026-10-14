"""Release participant IDs (p001...) and the private mapping back to `site_pNNN` keys.

`site_pNNN` keys derive from hospital screening numbers, so the release renumbers
participants. The mapping is written once to the private directory (mode 600, outside
the release tree) and reused by every later build so IDs stay stable.
"""

from __future__ import annotations

import csv
import io
import random
import re
import secrets
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .constants import ID_ORDERS, PRIVATE_FILE_MODE
from .safety import PathPolicy, atomic_write_bytes

RELEASE_ID_RE = re.compile(r"^p\d{3}$")
MAPPING_HEADER = ("participant_key", "release_id")
MAX_PARTICIPANTS = 999


def assign_release_ids(keys: Iterable[str], order: str = "random", seed: Optional[int] = None) -> Dict[str, str]:
    """Number participants p001..pNNN, either shuffled or in (site, number) order."""
    keys = list(keys)
    unique = sorted(set(keys))
    if len(unique) != len(keys):
        raise ValueError("duplicate participant keys")
    if len(unique) > MAX_PARTICIPANTS:
        raise ValueError(f"at most {MAX_PARTICIPANTS} participants fit the pNNN format")
    if order not in ID_ORDERS:
        raise ValueError(f"order must be one of {ID_ORDERS}, got {order!r}")
    ordered = list(unique)
    if order == "random":
        rng = random.Random(seed) if seed is not None else secrets.SystemRandom()
        rng.shuffle(ordered)
    return {key: f"p{i:03d}" for i, key in enumerate(ordered, start=1)}


def mapping_to_csv(mapping: Dict[str, str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MAPPING_HEADER)
    for key, release_id in sorted(mapping.items(), key=lambda item: item[1]):
        writer.writerow([key, release_id])
    return buffer.getvalue()


def parse_mapping_csv(text: str) -> Dict[str, str]:
    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    if not rows or tuple(rows[0]) != MAPPING_HEADER:
        raise ValueError(f"mapping file must start with header {MAPPING_HEADER}")
    mapping: Dict[str, str] = {}
    used_ids = set()
    for row in rows[1:]:
        if len(row) != 2:
            raise ValueError(f"malformed mapping row {row}")
        key, release_id = row
        if not RELEASE_ID_RE.match(release_id):
            raise ValueError(f"invalid release id {release_id!r}")
        if key in mapping:
            raise ValueError(f"duplicate participant key {key}")
        if release_id in used_ids:
            raise ValueError(f"duplicate release id {release_id}")
        mapping[key] = release_id
        used_ids.add(release_id)
    return mapping


def load_mapping(path, keys: Iterable[str]) -> Dict[str, str]:
    """Read an existing mapping (never writes) and check it numbers exactly `keys` as p001..pNNN."""
    path = Path(path)
    wanted = set(keys)
    mapping = parse_mapping_csv(path.read_text(encoding="utf-8"))
    if set(mapping) != wanted:
        missing = sorted(wanted - set(mapping))[:5]
        extra = sorted(set(mapping) - wanted)[:5]
        raise ValueError(
            f"existing mapping {path} does not match the released participants "
            f"(missing {missing}, extra {extra}); refusing to renumber"
        )
    if set(mapping.values()) != {f"p{i:03d}" for i in range(1, len(wanted) + 1)}:
        raise ValueError(f"existing mapping {path} is not a contiguous p001..p{len(wanted):03d} numbering")
    return mapping


def load_or_create_mapping(
    policy: PathPolicy,
    path,
    keys: Iterable[str],
    order: str = "random",
    seed: Optional[int] = None,
) -> Tuple[Dict[str, str], bool]:
    """Return (mapping, created). An existing mapping always wins and is never renumbered."""
    path = Path(path)
    if path.exists():
        return load_mapping(path, keys), False
    mapping = assign_release_ids(sorted(set(keys)), order=order, seed=seed)
    atomic_write_bytes(policy, path, mapping_to_csv(mapping).encode("utf-8"), PRIVATE_FILE_MODE)
    return mapping, True

"""Write guards that keep the source dataset (<source_root>) read-only.

Every file or directory the builder creates goes through a `PathPolicy`:
- writes must land below a configured write root and never below the source root or
  any extra protected root (--protect / KRRPPG_PROTECTED_ROOTS, e.g. the whole folder that holds
  <source_root>; none by default);
- output paths may not contain symlinks (realpath must equal abspath), so a link
  planted in the output tree cannot redirect a write, rename or unlink;
- pre-existing directories are never chmod-ed; new ones are created with their final
  mode (no chmod), so an inherited setgid bit - and the parent's group - survives on NFS.
The builder never sets a group itself: which Unix group owns the output is chosen by the operator through
the group and setgid bit of the output parent folder (no group name is built in).
Source files are only ever opened for reading.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import os
import socket
import stat
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

Signature = Optional[Tuple[str, int, int, int, int, int]]


class UnsafePathError(RuntimeError):
    """A write would land inside a protected tree, outside the output roots, or via a symlink."""


class BuildLockedError(RuntimeError):
    """Another build holds the lock on the same private directory."""


def resolve(path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def is_within(path, root) -> bool:
    target, base = resolve(path), resolve(root)
    return target == base or base in target.parents


class PathPolicy:
    """Allows writes only below `write_roots`, never below the source or protected roots."""

    def __init__(self, source_root, write_roots: Iterable, protected_roots: Iterable = ()):
        self.source_root = resolve(source_root)
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source root {self.source_root} is not a directory")
        self.protected_roots = (self.source_root,) + tuple(resolve(p) for p in protected_roots)
        self.write_roots = tuple(resolve(root) for root in write_roots)
        if not self.write_roots:
            raise ValueError("at least one write root is required")
        for root in self.write_roots:
            for protected in self.protected_roots:
                if is_within(root, protected) or is_within(protected, root):
                    raise UnsafePathError(f"write root {root} overlaps the protected tree {protected}")
        for a, b in itertools.combinations(self.write_roots, 2):
            if is_within(a, b) or is_within(b, a):
                raise UnsafePathError(f"write roots {a} and {b} must not be nested")

    def check_writable(self, path) -> Path:
        target = resolve(path)
        for protected in self.protected_roots:
            if is_within(target, protected):
                raise UnsafePathError(f"refusing to write inside the protected tree {protected}: {target}")
        if not any(is_within(target, root) for root in self.write_roots):
            raise UnsafePathError(f"refusing to write outside the output roots: {target}")
        if Path(os.path.abspath(os.fspath(path))) != target:
            raise UnsafePathError(f"refusing to write through a symlink: {path} -> {target}")
        return target

    def check_source(self, path) -> Path:
        target = resolve(path)
        if not is_within(target, self.source_root):
            raise UnsafePathError(f"{target} is not inside the source root {self.source_root}")
        return target

    def root_of(self, path) -> Path:
        target = self.check_writable(path)
        return next(root for root in self.write_roots if is_within(target, root))


def _mkdir_with_mode(directory: Path, mode: int) -> None:
    """mkdir with exactly `mode` and no chmod afterwards.

    On some NFS mounts a chmod clears the setgid bit a new directory inherits from its
    parent (so files would lose the parent folder's group). Creating the directory with the
    final mode under umask 0 keeps the inherited setgid bit.
    """
    previous = os.umask(0)
    try:
        directory.mkdir(mode=mode)
    finally:
        os.umask(previous)


def ensure_root(policy: PathPolicy, root, mode: int) -> Path:
    """Create a configured write root (parent must exist). Existing roots are left as they are."""
    target = resolve(root)
    if target not in policy.write_roots:
        raise UnsafePathError(f"{target} is not a configured write root")
    if target.exists():
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        return target
    if not target.parent.is_dir():
        raise FileNotFoundError(f"parent directory of {target} does not exist")
    policy.check_writable(target)
    _mkdir_with_mode(target, mode)
    return target


def make_dirs(policy: PathPolicy, path, mode: int) -> Path:
    """mkdir -p below a write root, applying `mode` only to the directories it creates."""
    target = policy.check_writable(path)
    root = policy.root_of(target)
    if not root.is_dir():
        raise FileNotFoundError(f"write root {root} does not exist; call ensure_root first")
    missing: List[Path] = []
    current = target
    while current != root and not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        try:
            _mkdir_with_mode(directory, mode)
        except FileExistsError:
            continue
    if not target.is_dir():
        raise NotADirectoryError(str(target))
    return target


def atomic_write_bytes(policy: PathPolicy, path, data: bytes, mode: int) -> Path:
    """Write via a hidden temp file in the same directory, fsync, then rename into place."""
    target = policy.check_writable(path)
    policy.check_writable(target.parent)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".partial")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        replace_checked(policy, tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    return target


def replace_checked(policy: PathPolicy, tmp, target) -> None:
    """os.replace with both paths re-validated immediately before the rename."""
    policy.check_writable(tmp)
    policy.check_writable(target)
    os.replace(tmp, target)


def copy_file(policy: PathPolicy, src, dst, mode: int) -> Path:
    """Copy bytes from a (read-only) source file to a guarded destination."""
    with open(src, "rb") as fh:
        data = fh.read()
    return atomic_write_bytes(policy, dst, data, mode)


def remove_output_file(policy: PathPolicy, path) -> None:
    """Delete a file the builder itself created (never valid for protected paths or symlinks)."""
    target = policy.check_writable(path)
    with contextlib.suppress(FileNotFoundError):
        target.unlink()


@contextlib.contextmanager
def build_lock(policy: PathPolicy, path) -> Iterator[Path]:
    """Exclusive O_EXCL lock file (NFS-safe); a stale lock must be removed by hand."""
    target = policy.check_writable(path)
    info = {"pid": os.getpid(), "host": socket.gethostname(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        holder = target.read_text(encoding="utf-8", errors="replace").strip()
        raise BuildLockedError(
            f"{target} exists ({holder}); another build is running or crashed - remove it if stale"
        ) from exc
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(info))
    try:
        yield target
    finally:
        with contextlib.suppress(FileNotFoundError):
            target.unlink()


def sha256_file(path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path) -> Signature:
    """(type, mode, size, mtime_ns, ctime_ns, inode) from lstat, or None if missing."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    kind = "d" if stat.S_ISDIR(st.st_mode) else "l" if stat.S_ISLNK(st.st_mode) else "f"
    return (kind, stat.S_IMODE(st.st_mode), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)


def snapshot(paths: Iterable) -> Dict[str, Signature]:
    return {str(path): file_signature(path) for path in paths}


def snapshot_tree(roots: Iterable) -> Dict[str, Signature]:
    """Signatures of each root and of every entry below it (symlinks are not followed)."""
    out: Dict[str, Signature] = {}
    for root in roots:
        out[str(root)] = file_signature(root)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in dirnames + filenames:
                path = os.path.join(dirpath, name)
                out[path] = file_signature(path)
    return out


def diff_snapshots(before: Dict[str, Signature], after: Dict[str, Signature]) -> List[str]:
    changes = [f"{path}: {sig} -> {after.get(path)}" for path, sig in before.items() if after.get(path) != sig]
    changes += [f"{path}: added" for path in after if path not in before]
    return changes

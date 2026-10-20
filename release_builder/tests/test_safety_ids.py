import os
import stat

import pytest

from conftest import chmod_or_skip, on_network_filesystem
from krrppg_release.ids import (
    assign_release_ids,
    load_or_create_mapping,
    mapping_to_csv,
    parse_mapping_csv,
)
from krrppg_release.safety import (
    BuildLockedError,
    PathPolicy,
    UnsafePathError,
    atomic_write_bytes,
    build_lock,
    copy_file,
    diff_snapshots,
    ensure_root,
    make_dirs,
    remove_output_file,
    snapshot,
    snapshot_tree,
)


@pytest.fixture
def roots(tmp_path):
    src = tmp_path / "source_root"
    (src / "a").mkdir(parents=True)
    (src / "a" / "f.txt").write_text("x")
    return src, tmp_path / "out", tmp_path / "private"


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


# --- PathPolicy -------------------------------------------------------------

def test_write_root_inside_source_is_rejected(roots):
    src, _out, _priv = roots
    with pytest.raises(UnsafePathError):
        PathPolicy(src, [src / "release"])


def test_source_inside_write_root_is_rejected(roots, tmp_path):
    src, _out, _priv = roots
    with pytest.raises(UnsafePathError):
        PathPolicy(src, [tmp_path])


def test_nested_write_roots_are_rejected(roots):
    src, out, _priv = roots
    with pytest.raises(UnsafePathError):
        PathPolicy(src, [out, out / "private"])


def test_only_paths_below_write_roots_are_writable(roots, tmp_path):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    policy.check_writable(out / "p001" / "T00" / "x.csv")
    policy.check_writable(priv / "map.csv")
    with pytest.raises(UnsafePathError):
        policy.check_writable(src / "a" / "f.txt")
    with pytest.raises(UnsafePathError):
        policy.check_writable(tmp_path / "elsewhere.txt")


def test_symlink_from_output_into_source_is_rejected(roots):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    out.mkdir()
    (out / "link").symlink_to(src / "a")
    with pytest.raises(UnsafePathError):
        policy.check_writable(out / "link" / "g.txt")


def test_check_source_requires_paths_inside_source(roots, tmp_path):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    policy.check_source(src / "a" / "f.txt")
    with pytest.raises(UnsafePathError):
        policy.check_source(tmp_path / "other")


# --- write helpers ------------------------------------------------------------

def test_atomic_write_sets_mode_and_leaves_no_temp_files(roots):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    ensure_root(policy, out, 0o750)
    target = out / "f.csv"
    atomic_write_bytes(policy, target, b"abc", 0o640)
    assert target.read_bytes() == b"abc"
    assert _mode(target) == 0o640
    assert _mode(out) & 0o777 == 0o750
    # a setgid bit inherited from the temp parent is the filesystem's doing, not the builder's
    assert _mode(out) & stat.S_ISGID == _mode(out.parent) & stat.S_ISGID
    assert os.listdir(out) == ["f.csv"]


def test_atomic_write_into_source_is_refused(roots):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    with pytest.raises(UnsafePathError):
        atomic_write_bytes(policy, src / "a" / "f.txt", b"overwrite", 0o640)
    assert (src / "a" / "f.txt").read_text() == "x"


def test_copy_reads_source_without_changing_it(roots):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    ensure_root(policy, out, 0o750)
    watched = [src / "a" / "f.txt"]
    before = snapshot(watched)
    copy_file(policy, src / "a" / "f.txt", out / "f.txt", 0o640)
    assert (out / "f.txt").read_text() == "x"
    assert diff_snapshots(before, snapshot(watched)) == []


def test_snapshot_diff_reports_modified_and_missing_files(roots):
    src, _out, _priv = roots
    path = src / "a" / "f.txt"
    before = snapshot([path])
    os.utime(path, ns=(1, 1))
    assert diff_snapshots(before, snapshot([path]))
    path.unlink()
    assert diff_snapshots(before, snapshot([path]))


def test_ensure_root_requires_existing_parent(roots, tmp_path):
    src, _out, _priv = roots
    deep = tmp_path / "missing_parent" / "out"
    policy = PathPolicy(src, [deep])
    with pytest.raises(FileNotFoundError):
        ensure_root(policy, deep, 0o750)


def test_existing_root_keeps_its_permissions(roots):
    src, out, priv = roots
    out.mkdir()
    chmod_or_skip(out, 0o2775)
    policy = PathPolicy(src, [out, priv])
    ensure_root(policy, out, 0o750)
    assert _mode(out) == 0o2775


def test_new_directories_keep_an_inherited_setgid_bit(roots, tmp_path):
    src, out, priv = roots
    chmod_or_skip(tmp_path, 0o2775)
    policy = PathPolicy(src, [out, priv])
    ensure_root(policy, out, 0o750)
    made = make_dirs(policy, out / "KR-RPPG" / "p001", 0o750)
    assert _mode(out) == 0o2750
    assert _mode(made) == 0o2750


def test_new_directories_are_never_chmod_ed(roots, monkeypatch):
    # On some NFS mounts a chmod strips the inherited setgid bit, so dirs must be created with their final mode.
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    chmodded = []
    real_chmod = os.chmod
    monkeypatch.setattr(os, "chmod", lambda path, mode, *a, **k: (chmodded.append(str(path)), real_chmod(path, mode, *a, **k)))
    ensure_root(policy, out, 0o750)
    made = make_dirs(policy, out / "KR-RPPG" / "p001" / "T00", 0o750)
    assert chmodded == []
    assert _mode(made) & 0o777 == 0o750


def test_writing_through_a_symlinked_file_is_refused(roots):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    ensure_root(policy, out, 0o750)
    ensure_root(policy, priv, 0o700)
    (priv / "participant_id_map.csv").write_text("secret")
    (out / ".stale.partial").symlink_to(priv / "participant_id_map.csv")
    with pytest.raises(UnsafePathError):
        remove_output_file(policy, out / ".stale.partial")
    with pytest.raises(UnsafePathError):
        atomic_write_bytes(policy, out / ".stale.partial", b"x", 0o640)
    assert (priv / "participant_id_map.csv").read_text() == "secret"


def test_protected_roots_cover_siblings_of_the_source(tmp_path):
    dataset = tmp_path / "dataset_folder"
    (dataset / "source_root").mkdir(parents=True)
    (dataset / "figures").mkdir()
    with pytest.raises(UnsafePathError):
        PathPolicy(dataset / "source_root", [dataset / "figures" / "out"], protected_roots=[dataset])
    with pytest.raises(UnsafePathError):
        PathPolicy(dataset / "source_root", [tmp_path], protected_roots=[dataset])
    PathPolicy(dataset / "source_root", [tmp_path / "out"], protected_roots=[dataset])


def test_build_lock_is_exclusive_and_released(roots):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    ensure_root(policy, priv, 0o700)
    lock = priv / "build.lock"
    with build_lock(policy, lock):
        with pytest.raises(BuildLockedError):
            with build_lock(policy, lock):
                pass
    assert not lock.exists()


def test_tree_snapshot_sees_new_files_and_permission_changes(roots, request):
    src, _out, _priv = roots
    if on_network_filesystem(src):
        # Seen on NFS: the client's attribute/directory cache can still return the old listing and
        # stat right after the change, so the second snapshot may equal the first. Deterministic on local disks.
        request.applymarker(pytest.mark.xfail(
            reason="network filesystem: client-side attribute caching can hide a change made a moment earlier",
            strict=False))
    before = snapshot_tree([src])
    (src / "a" / ".video.partial.avi").write_bytes(b"")
    assert diff_snapshots(before, snapshot_tree([src]))
    (src / "a" / ".video.partial.avi").unlink()
    os.chmod(src / "a", 0o755 if _mode(src / "a") != 0o755 else 0o700)
    changes = diff_snapshots(before, snapshot_tree([src]))
    assert any(change.startswith(str(src / "a") + ":") for change in changes)


def test_make_dirs_is_refused_in_source(roots):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    with pytest.raises(UnsafePathError):
        make_dirs(policy, src / "new_dir", 0o750)
    assert not (src / "new_dir").exists()


# --- release IDs --------------------------------------------------------------

KEYS = ["A_p903", "C_p902", "C_p901"]


def test_sorted_order_assigns_ids_by_site_then_number():
    assert assign_release_ids(KEYS, order="sorted") == {
        "A_p903": "p001",
        "C_p901": "p002",
        "C_p902": "p003",
    }


def test_random_order_is_a_seeded_bijection():
    keys = [f"C_p{i:03d}" for i in range(1, 51)]
    first = assign_release_ids(keys, order="random", seed=7)
    assert first == assign_release_ids(keys, order="random", seed=7)
    assert set(first.values()) == {f"p{i:03d}" for i in range(1, 51)}
    assert first != assign_release_ids(keys, order="sorted")


def test_duplicate_keys_and_unknown_order_are_rejected():
    with pytest.raises(ValueError):
        assign_release_ids(["a_p001", "a_p001"], order="sorted")
    with pytest.raises(ValueError):
        assign_release_ids(KEYS, order="alphabetical")


def test_mapping_csv_round_trip():
    mapping = assign_release_ids(KEYS, order="random", seed=1)
    assert parse_mapping_csv(mapping_to_csv(mapping)) == mapping


@pytest.mark.parametrize(
    "text",
    [
        "participant_key,release_id\na_p001,p001\nb_p002,p001\n",
        "participant_key,release_id\na_p001,p001\na_p001,p002\n",
        "participant_key,release_id\na_p001,x001\n",
        "key,id\na_p001,p001\n",
    ],
)
def test_malformed_mapping_is_rejected(text):
    with pytest.raises(ValueError):
        parse_mapping_csv(text)


def test_mapping_is_created_once_and_then_reused(roots):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    ensure_root(policy, priv, 0o700)
    path = priv / "participant_id_map.csv"
    first, created = load_or_create_mapping(policy, path, KEYS, order="random", seed=None)
    assert created
    assert _mode(path) == 0o600
    second, created_again = load_or_create_mapping(policy, path, KEYS, order="sorted", seed=None)
    assert not created_again
    assert second == first


def test_existing_mapping_with_different_participants_is_rejected(roots):
    src, out, priv = roots
    policy = PathPolicy(src, [out, priv])
    ensure_root(policy, priv, 0o700)
    path = priv / "participant_id_map.csv"
    load_or_create_mapping(policy, path, KEYS, order="sorted", seed=None)
    with pytest.raises(ValueError):
        load_or_create_mapping(policy, path, KEYS + ["A_p905"], order="sorted", seed=None)

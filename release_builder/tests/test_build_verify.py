import csv
import json
import os
import shutil
import stat

import cv2
import numpy as np
import pytest

from conftest import EPOCH_T0, FRAME_H, FRAME_W, PARTICIPANTS, chmod_or_skip, make_video, tree_fingerprint
from krrppg_release import __version__
from krrppg_release import constants as C
from krrppg_release.bfdt import release_keep_ratio
from krrppg_release.builder import BuildConfig, build_release, refresh_metadata
from krrppg_release.clinical import NUMERIC_COLUMNS, PARTICIPANT_COLUMNS
from krrppg_release.ownership import ReleaseConflictError
from krrppg_release.cli import main
from krrppg_release.ids import parse_mapping_csv
from krrppg_release.safety import BuildLockedError, PathPolicy, UnsafePathError
from krrppg_release.verify import VerifyReport, _check_participants, verify_release

SIZE = (FRAME_W, FRAME_H)


def _config(fake_source, **overrides):
    root = fake_source["root"]
    work = fake_source["workdir"]
    params = dict(
        source_root=root,
        metadata_dir=root / "metadata",
        output_dir=work / "release_out",
        private_dir=work / "release_private",
        ild_xlsx=fake_source["ild_xlsx"],
        seed=3,
        workers=1,
        expected_frame_size=SIZE,
    )
    params.update(overrides)
    return BuildConfig(**params)


def _mapping(cfg):
    return parse_mapping_csv(cfg.mapping_path.read_text())


def _verify(cfg, **kwargs):
    kwargs.setdefault("expected_frame_size", SIZE)
    kwargs.setdefault("ild_xlsx", cfg.ild_xlsx)
    return verify_release(cfg.release_root, **kwargs)


def test_build_creates_figure2_layout_without_touching_source(fake_source):
    root = fake_source["root"]
    before = tree_fingerprint(root)
    cfg = _config(fake_source)

    report = build_release(cfg)

    assert tree_fingerprint(root) == before
    assert report["source_unchanged"] is True
    assert (report["built"], report["skipped"], report["failed"]) == (4, 0, 0)
    release = cfg.release_root
    assert release.name == "KR-rPPG"
    assert sorted(os.listdir(release)) == ["metadata", "p001", "p002", "p003"]

    ids = _mapping(cfg)
    assert sorted(os.listdir(release / ids["C_p901"])) == ["T00", "T01"]
    assert os.listdir(release / ids["C_p902"]) == ["T00"]  # T01 exists in source but is not released
    assert os.listdir(release / ids["A_p903"]) == ["T02"]
    for pid in ("p001", "p002", "p003"):
        for task in os.listdir(release / pid):
            assert sorted(os.listdir(release / pid / task)) == sorted(C.RELEASE_SESSION_FILES)

    assert not str(cfg.mapping_path).startswith(str(cfg.output_dir))
    assert stat.S_IMODE(os.stat(cfg.mapping_path).st_mode) == 0o600
    result = _verify(cfg)
    assert result.errors == [], result.errors
    assert result.sessions == 4


def test_output_inside_source_is_refused_before_any_write(fake_source):
    root = fake_source["root"]
    before = tree_fingerprint(root)
    with pytest.raises(UnsafePathError):
        build_release(_config(fake_source, output_dir=root / "release"))
    assert tree_fingerprint(root) == before


def test_private_dir_inside_output_is_refused(fake_source):
    cfg = _config(fake_source)
    with pytest.raises(UnsafePathError):
        build_release(_config(fake_source, private_dir=cfg.output_dir / "private"))


def test_participant_subset_builds_only_selected_sessions(fake_source):
    cfg = _config(fake_source, participants=("A_p903",))
    report = build_release(cfg)
    assert (report["built"], report["failed"]) == (1, 0)
    assert _verify(cfg, partial=True).errors == []
    assert _verify(cfg, partial=False).errors  # full-release check notices missing sessions


def test_rerun_skips_completed_sessions(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    second = build_release(cfg)
    assert (second["built"], second["skipped"], second["failed"]) == (0, 4, 0)


def test_process_pool_build_matches_serial_build(fake_source):
    serial = _config(fake_source)
    pooled = _config(
        fake_source,
        output_dir=fake_source["workdir"] / "pooled_out",
        private_dir=fake_source["workdir"] / "pooled_private",
        workers=2,
    )
    build_release(serial)
    report = build_release(pooled)
    assert report["failed"] == 0
    for pid in ("p001", "p002", "p003"):
        for task in os.listdir(serial.release_root / pid):
            for name in C.RELEASE_SESSION_FILES:
                a = (serial.release_root / pid / task / name).read_bytes()
                b = (pooled.release_root / pid / task / name).read_bytes()
                assert a == b, (pid, task, name)


def test_bad_session_is_reported_and_others_still_build(fake_source):
    bvp = fake_source["root"] / "A" / "01-groupa" / "p903" / "T02" / C.BVP_SYNC
    lines = bvp.read_text().splitlines()
    bvp.write_text("\n".join(lines[:-1]) + "\n")  # synthetic source only
    cfg = _config(fake_source)
    report = build_release(cfg)
    assert (report["built"], report["failed"]) == (3, 1)
    assert "BVP_synchronized.csv" in report["failures"][0]["error"]
    failed_dir = cfg.release_root / _mapping(cfg)["A_p903"] / "T02"
    assert not (failed_dir / C.RELEASE_VIDEO).exists()


def test_dry_run_writes_nothing(fake_source):
    cfg = _config(fake_source, dry_run=True)
    report = build_release(cfg)
    assert report["sessions_total"] == 4
    assert not cfg.output_dir.exists()
    assert not cfg.private_dir.exists()


# --- verifier catches tampering -------------------------------------------------

def _first_session(cfg):
    pid = sorted(p for p in os.listdir(cfg.release_root) if p.startswith("p"))[0]
    task = sorted(os.listdir(cfg.release_root / pid))[0]
    return cfg.release_root / pid / task


def _stray_file(session, release):
    (session / "notes.txt").write_text("x")


def _raw_video(session, release):
    rng = np.random.default_rng(0)
    video = session / C.RELEASE_VIDEO
    video.unlink()
    make_video(video, [rng.integers(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8) for _ in range(6)])


def _epoch_timestamps(session, release):
    path = session / C.FRAMES_TIMESTAMP
    rows = list(csv.reader(path.read_text().splitlines()))
    rows[1][1] = EPOCH_T0
    with path.open("w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)


def _old_identifier(session, release):
    path = release / "metadata" / "fold_assignment.csv"
    path.write_text(path.read_text() + "C_p901,1\n")


def _partial_leftover(session, release):
    (release / ".video_bfdt_ffv1.123.partial.avi").write_bytes(b"")


def _info_txt_left_behind(session, release):
    (session / "info.txt").write_text("cam model: Brio 500\ncam codec: YUY2\nsize: 64x48\n")


def _site_in_folds(session, release):
    path = release / "metadata" / "fold_assignment.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0] + ",site"] + [line + ",A" for line in lines[1:]]) + "\n")


def _rewrite_participants(release, edit):
    path = release / "metadata" / "participants.csv"
    rows = list(csv.reader(path.read_text().splitlines()))
    edit(rows)
    with path.open("w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)


def _extra_column_in_participants(session, release):
    _rewrite_participants(release, lambda rows: [row.append("x") for row in rows])


def _age_band_in_participants(session, release):
    def edit(rows):
        rows[1][PARTICIPANT_COLUMNS.index("age")] = "60-69"  # an age band instead of the age in years
    _rewrite_participants(release, edit)


def _columns_reordered_in_participants(session, release):
    def edit(rows):
        for row in rows:
            row[2], row[3] = row[3], row[2]  # age <-> sex
    _rewrite_participants(release, edit)


def _set_recorded_cell(release, column, value):
    """Replace the first non-blank `column` cell (so the dictionary availability count stays right)."""
    index = PARTICIPANT_COLUMNS.index(column)

    def edit(rows):
        row = next(row for row in rows[1:] if row[index] != "")
        row[index] = value
    _rewrite_participants(release, edit)


def _off_step_sbp_in_participants(session, release):
    _set_recorded_cell(release, "sbp", "123")  # not a multiple of 5


def _unrounded_creatinine_in_participants(session, release):
    _set_recorded_cell(release, "creatinine", "0.93")  # two decimals


def _hba1c_without_decimal_in_participants(session, release):
    _set_recorded_cell(release, "hba1c", "7")  # hba1c is written with one decimal


def _category_in_rounded_column(session, release):
    _set_recorded_cell(release, "sbp", "120-129")  # a category label instead of the number


def _fractional_age_in_participants(session, release):
    _set_recorded_cell(release, "age", "66.5")  # age is a whole number of years


def _off_step_height_in_participants(session, release):
    _set_recorded_cell(release, "height", "172")  # not a multiple of 5


def _odd_weight_in_participants(session, release):
    _set_recorded_cell(release, "weight", "71")  # not a multiple of 2


def _height_in_metres_in_participants(session, release):
    _set_recorded_cell(release, "height", "0")  # on the grid but below the plausible range (unit mix-up)


def _bmi_not_from_height_and_weight(session, release):
    _set_recorded_cell(release, "bmi", "99.9")  # one decimal, but not weight / (height / 100)^2


def _bmi_with_two_decimals(session, release):
    _set_recorded_cell(release, "bmi", "24.22")


def _top_coded_comorbidity_count(session, release):
    _set_recorded_cell(release, "other_comorbidity_count", ">=1")  # the former top-coding


def _comorbidity_count_above_five(session, release):
    _set_recorded_cell(release, "other_comorbidity_count", "6")  # five indicators at most


def _implausible_fpg_in_participants(session, release):
    _set_recorded_cell(release, "fpg", "0")  # on the 10 mg/dL grid but below the plausible range (10 since 1.4.0)


def _implausible_sbp_in_participants(session, release):
    _set_recorded_cell(release, "sbp", "45")  # on the 5 mmHg grid but below the plausible range (50 since 1.4.0)


def _measurement_blanked_without_dictionary(session, release):
    _set_recorded_cell(release, "gfr", "")


def _dictionary_description_edited(session, release):
    path = release / "metadata" / "participants_dictionary.csv"
    path.write_text(path.read_text().replace("Interstitial lung disease", "Acute lung disease"))


def _dictionary_availability_edited(session, release):
    path = release / "metadata" / "participants_dictionary.csv"
    lines = path.read_text().splitlines()
    lines[-1] = lines[-1].rsplit(",", 1)[0] + ",999"
    path.write_text("\n".join(lines) + "\n")


def _nul_byte_in_csv(session, release):
    path = session / C.HR_SYNC
    path.write_bytes(path.read_bytes() + b"\x00garbage\n")


@pytest.mark.parametrize(
    "tamper",
    [_stray_file, _raw_video, _epoch_timestamps, _old_identifier, _partial_leftover, _info_txt_left_behind,
     _nul_byte_in_csv, _extra_column_in_participants, _age_band_in_participants, _columns_reordered_in_participants,
     _dictionary_description_edited, _dictionary_availability_edited, _site_in_folds,
     _off_step_sbp_in_participants, _unrounded_creatinine_in_participants, _hba1c_without_decimal_in_participants,
     _category_in_rounded_column, _implausible_fpg_in_participants, _implausible_sbp_in_participants,
     _measurement_blanked_without_dictionary, _fractional_age_in_participants, _off_step_height_in_participants,
     _odd_weight_in_participants, _height_in_metres_in_participants, _bmi_not_from_height_and_weight,
     _bmi_with_two_decimals, _top_coded_comorbidity_count, _comorbidity_count_above_five],
)
def test_verifier_detects_tampering(fake_source, tamper):
    cfg = _config(fake_source)
    build_release(cfg)
    assert _verify(cfg).errors == []
    tamper(_first_session(cfg), cfg.release_root)
    assert _verify(cfg).errors


def test_source_cross_check_detects_value_drift(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    clean = _verify(cfg, source_root=fake_source["root"], mapping_path=cfg.mapping_path)
    assert clean.errors == []
    bvp = _first_session(cfg) / C.BVP_SYNC
    rows = list(csv.reader(bvp.read_text().splitlines()))
    rows[1][1] = "123.0"
    with bvp.open("w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    drift = _verify(cfg, source_root=fake_source["root"], mapping_path=cfg.mapping_path)
    assert any("differs from source" in e for e in drift.errors)


def test_parallel_verification_matches_serial(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    (_first_session(cfg) / "notes.txt").write_text("x")  # one error to compare
    kwargs = dict(source_root=fake_source["root"], mapping_path=cfg.mapping_path)
    serial = _verify(cfg, **kwargs)
    parallel = _verify(cfg, workers=2, **kwargs)
    assert serial.errors and parallel.errors == serial.errors
    assert (parallel.sessions, parallel.frames, parallel.warnings) == (serial.sessions, serial.frames, serial.warnings)


def test_source_cross_check_detects_metadata_tampering(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    age = PARTICIPANT_COLUMNS.index("age")

    def swap_ages(rows):
        rows[1][age], rows[2][age] = rows[2][age], rows[1][age]  # valid ages, wrong participants

    _rewrite_participants(cfg.release_root, swap_ages)
    assert _verify(cfg).errors == []  # still valid ages and dictionary counts: only the source render sees it
    result = _verify(cfg, source_root=fake_source["root"], mapping_path=cfg.mapping_path)
    assert result.errors == ["metadata/participants.csv differs from source"]


def test_verifier_never_quotes_participant_values(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    _age_band_in_participants(None, cfg.release_root)
    _set_recorded_cell(cfg.release_root, "sex", "X")
    errors = _verify(cfg).errors
    assert errors == ["participants.csv: 1 age cells are not whole numbers written without decimals (blank = missing)",
                      f"participants.csv: 1 sex cells hold a value outside {['M', 'F']} (blank = missing)"]
    assert not any("60-69" in e or "'X'" in e for e in errors)


def test_verifier_accepts_only_site_letters(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    _set_recorded_cell(cfg.release_root, "site", "B")  # another site letter: valid
    assert _verify(cfg).errors == []
    _set_recorded_cell(cfg.release_root, "site", "Site-X")  # anything else (e.g. a site name) is an error
    errors = _verify(cfg).errors
    assert errors == [f"participants.csv: 1 site cells hold a value outside {['A', 'B', 'C']} (blank = missing)"]
    assert not any("Site-X" in e for e in errors)  # the value is never quoted


def test_verifier_never_quotes_rounded_values(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    _set_recorded_cell(cfg.release_root, "sbp", "123")
    _set_recorded_cell(cfg.release_root, "hba1c", "7.25")
    _set_recorded_cell(cfg.release_root, "fpg", "3000")
    errors = _verify(cfg).errors
    assert errors == [
        "participants.csv: 1 sbp cells are not a multiple of 5 written with 0 decimals (blank = missing)",
        "participants.csv: 1 hba1c cells are not a multiple of 0.5 written with 1 decimals (blank = missing)",
        "participants.csv: 1 fpg cells lie outside the plausible range 10 to 1000",
    ]
    assert not any(value in e for e in errors for value in ("123", "7.25", "3000"))


def test_verifier_accepts_rounded_numbers_on_the_grid(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    for column, value in (("sbp", "70"), ("dbp", "200"), ("hr_rest", "36"), ("hba1c", "20.0"), ("fpg", "40"),
                          ("creatinine", "0.1"), ("gfr", "0"), ("ast", "1000"), ("alt", "5"), ("age", "120"),
                          ("age", "0")):
        _set_recorded_cell(cfg.release_root, column, value)
    assert _verify(cfg).errors == []


def test_verifier_checks_bmi_against_height_and_weight(fake_source):
    """bmi = weight / (height / 100)^2 of the same row, one decimal (released_bmi); blank when either is blank."""
    cfg = _config(fake_source)
    build_release(cfg)
    _set_recorded_cell(cfg.release_root, "height", "250")  # on the grid and in range, but bmi no longer matches
    message = ("participants.csv: 1 bmi cells differ from weight / (height / 100)^2 of the row's height and weight "
               "cells, one decimal (blank when either is blank)")
    assert _verify(cfg).errors == [message]
    path = cfg.release_root / "metadata" / "participants.csv"
    rows = list(csv.reader(path.read_text().splitlines()))
    height, weight, bmi = (PARTICIPANT_COLUMNS.index(c) for c in ("height", "weight", "bmi"))
    row = next(row for row in rows[1:] if row[height] == "250")
    row[bmi] = f"{round(float(row[weight]) / 2.5 ** 2, 1):.1f}"
    with path.open("w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    assert _verify(cfg).errors == []
    row[weight] = ""  # bmi without a weight
    with path.open("w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    assert message in _verify(cfg).errors


def test_verifier_accepts_values_below_the_former_blanking_thresholds(fake_source):
    """1.4.0 releases the values the SBP < 70 and FPG < 40 rules used to blank; verify must accept them."""
    cfg = _config(fake_source)
    build_release(cfg)
    for column, value in (("sbp", "50"), ("dbp", "20"), ("fpg", "10")):
        _set_recorded_cell(cfg.release_root, column, value)
    assert _verify(cfg).errors == []


def test_verifier_checks_the_availability_of_rounded_columns(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    _measurement_blanked_without_dictionary(None, cfg.release_root)
    assert _verify(cfg).errors == ["participants_dictionary.csv: rows for ['gfr'] differ from the builder's dictionary "
                                   "(description, coding or availability count)"]


def test_verifier_warns_about_small_categories(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    warnings = _verify(cfg).warnings  # 3 synthetic participants: every category is below 5
    assert any(w.startswith("participants.csv: sex categories with fewer than 5 participants") for w in warnings)
    assert not any("participant_key" in w for w in warnings)
    # the numeric columns (age, height, weight, bmi, the nine measurements, the count) have no categories:
    # no warning names them (nor quotes their values)
    small = [w for w in warnings if "categories with fewer than" in w]
    assert small and not any(f" {column} categories" in w for w in small for column in NUMERIC_COLUMNS)
    assert len(NUMERIC_COLUMNS) == 14 and {"age", "height", "weight", "bmi", "other_comorbidity_count"} <= \
        set(NUMERIC_COLUMNS)


# --- resume / stale outputs / locking ----------------------------------------------

def _released_labels(cfg):
    return {f"{rid}/{task}" for key, rid in _mapping(cfg).items()
            for task, flag in PARTICIPANTS[key][3].items() if flag}


def _source_drift_errors(cfg, fake_source):
    """'differs from source' errors for sessions that are currently released."""
    result = _verify(cfg, source_root=fake_source["root"], mapping_path=cfg.mapping_path)
    released = _released_labels(cfg)
    return [e for e in result.errors if "differs" in e and e.split(":")[0] in released]


def test_regenerated_mapping_never_reuses_another_participants_outputs(fake_source):
    build_release(_config(fake_source))
    cfg = _config(fake_source)
    cfg.mapping_path.unlink()  # private file only
    report = build_release(_config(fake_source, id_order="sorted"))
    assert report["built"] >= 1  # the sorted numbering differs from the seed-3 one
    assert not _released_labels(cfg) & set(report["stale_entries"])
    assert _source_drift_errors(cfg, fake_source) == []


def test_changed_source_file_triggers_a_rebuild(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    boxes = fake_source["root"] / "A" / "01-groupa" / "p903" / "T02" / C.FACE_BBOXES
    boxes.write_text(boxes.read_text().replace("8,8,40,40", "9,9,41,41"))  # synthetic source only
    report = build_release(cfg)
    assert (report["built"], report["skipped"]) == (1, 3)
    assert _source_drift_errors(cfg, fake_source) == []


def test_skipped_sessions_keep_their_warnings(fake_source):
    boxes = fake_source["root"] / "A" / "01-groupa" / "p903" / "T02" / C.FACE_BBOXES
    boxes.write_text("\n".join(boxes.read_text().splitlines()[:-1]) + "\n")
    cfg = _config(fake_source)
    first = build_release(cfg)
    second = build_release(cfg)
    assert second["skipped"] == 4
    assert second["warnings"] == first["warnings"] != []


def test_withdrawn_session_is_reported_as_stale(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    flags = fake_source["root"] / "metadata" / "task_session_availability.csv"
    flags.write_text(flags.read_text().replace("C_p901,1,1,0,0", "C_p901,1,0,0,0"))
    report = build_release(cfg)
    assert report["stale_entries"] == [f"{_mapping(cfg)['C_p901']}/T01"]


def test_concurrent_build_is_refused(fake_source):
    cfg = _config(fake_source)
    cfg.private_dir.mkdir()
    cfg.lock_path.write_text('{"pid": 1}')
    with pytest.raises(BuildLockedError):
        build_release(cfg)
    assert not cfg.release_root.exists()


def test_existing_output_dir_permissions_are_left_alone(fake_source):
    cfg = _config(fake_source)
    cfg.output_dir.mkdir()
    chmod_or_skip(cfg.output_dir, 0o2775)
    build_release(cfg)
    assert stat.S_IMODE(os.stat(cfg.output_dir).st_mode) == 0o2775


def test_site_ordered_numbering_is_flagged(fake_source):
    cfg = _config(fake_source, id_order="sorted")
    build_release(cfg)
    assert any("follow site order" in w for w in _verify(cfg).warnings)


@pytest.mark.parametrize("sites, flagged", [(("C", "C", "B", "A"), True), (("A", "A", "C", "C"), True),
                                             (("A", "C", "A", "C"), False), (("C", "C", "C", "C"), False)])
def test_site_order_warning_needs_one_block_per_site(sites, flagged):
    report = VerifyReport()
    header = list(PARTICIPANT_COLUMNS)
    rows = [[f"p{i:03d}", site] + [""] * (len(header) - 2) for i, site in enumerate(sites, start=1)]
    _check_participants((header, rows), (["variable"], []), report)
    assert any("follow site order" in w for w in report.warnings) == flagged


def test_cli_does_not_offer_site_ordered_numbering(tmp_path):
    with pytest.raises(SystemExit):
        main(["build", "--output-dir", str(tmp_path / "x"), "--private-dir", str(tmp_path / "y"),
              "--id-order", "sorted"])


def test_sessions_hold_six_files_and_no_info_txt(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    sessions = sorted(cfg.release_root.glob("p*/T*"))
    assert sessions and all(sorted(os.listdir(s)) == sorted(C.RELEASE_SESSION_FILES) for s in sessions)
    assert len(C.RELEASE_SESSION_FILES) == 6 and "info.txt" not in C.RELEASE_SESSION_FILES
    assert _verify(cfg).errors == []


def test_build_writes_the_24_columns_and_nothing_outside_its_folders(fake_source):
    cfg = _config(fake_source)
    report = build_release(cfg)
    with open(cfg.release_root / "metadata" / "participants.csv", newline="") as fh:
        assert tuple(next(csv.reader(fh))) == PARTICIPANT_COLUMNS
    assert report["data_quality_rule_counts"]["height_outside_120_210_flagged_unchanged"] == 1
    assert not any(key.endswith("_set_missing") for key in report["data_quality_rule_counts"])
    assert sorted(os.listdir(fake_source["workdir"])) == ["ild_items.xlsx", "release_out", "release_private", "source_root"]


def test_protected_roots_are_refused(fake_source, tmp_path):
    """Pure path checks: a protected folder (e.g. another custodian folder) is never writable."""
    protected = str(tmp_path / "custodian" / "protected")
    policy = PathPolicy(fake_source["root"], [fake_source["workdir"] / "out"], (protected,))
    with pytest.raises(UnsafePathError):
        policy.check_writable(f"{protected}/participants.csv")
    with pytest.raises(UnsafePathError):  # a write root may not contain it either
        PathPolicy(fake_source["root"], [os.path.dirname(protected)], (protected,))


def test_protected_roots_come_from_the_environment(monkeypatch, tmp_path):
    import importlib

    roots = [str(tmp_path / "dataset"), str(tmp_path / "clinical")]
    monkeypatch.setenv("KRRPPG_PROTECTED_ROOTS", os.pathsep.join(roots))
    try:
        assert importlib.reload(C).PROTECTED_ROOTS == tuple(roots)
    finally:
        monkeypatch.delenv("KRRPPG_PROTECTED_ROOTS")
        importlib.reload(C)


def test_metadata_only_version_bump_keeps_session_records_valid(fake_source):
    assert __version__ == "1.5.0"
    cfg = _config(fake_source)
    report = build_release(cfg)
    assert report["builder_version"] == "1.5.0"
    assert cfg.settings()["builder_version"] == C.SESSION_OUTPUT_VERSION == "1.1.0"
    records = [json.loads(p.read_text()) for p in cfg.records_dir.glob("*.json")]
    assert records and all(r["settings"]["builder_version"] == "1.1.0" for r in records)
    assert build_release(cfg)["skipped"] == 4  # 1.1.0 session records are reused, nothing is re-encoded


# --- other block sizes -------------------------------------------------------------

K16 = release_keep_ratio(16)


def _config16(fake_source, **overrides):
    work = fake_source["workdir"]
    params = dict(output_dir=work / "out16", private_dir=work / "private16", block_size=16, keep_ratio=K16)
    params.update(overrides)
    return _config(fake_source, **params)


def test_16px_release_builds_and_matches_a_16px_render_of_the_source(fake_source):
    root = fake_source["root"]
    before = tree_fingerprint(root)
    cfg = _config16(fake_source)
    report = build_release(cfg)
    assert (report["built"], report["failed"]) == (4, 0)
    assert report["settings"]["block_size"] == 16 and report["settings"]["keep_ratio"] == K16
    assert tree_fingerprint(root) == before
    result = _verify(cfg, block_size=16, keep_ratio=K16, source_root=root, mapping_path=cfg.mapping_path)
    assert result.errors == [], result.errors


def _block_means_16(frame):
    """Independent of the package: rounded 16x16 means, tiled back to full size."""
    h, w, c = frame.shape
    means = np.rint(frame.reshape(h // 16, 16, w // 16, 16, c).astype(np.float64).mean(axis=(1, 3)))
    return np.repeat(np.repeat(means.astype(np.uint8), 16, axis=0), 16, axis=1)


def test_16px_video_pixels_are_the_16px_block_means_of_the_source(fake_source):
    cfg = _config16(fake_source)
    build_release(cfg)
    capture = cv2.VideoCapture(str(cfg.release_root / _mapping(cfg)["C_p901"] / "T00" / C.RELEASE_VIDEO))
    decoded = []
    ok, frame = capture.read()
    while ok:
        decoded.append(frame)
        ok, frame = capture.read()
    capture.release()
    source = fake_source["frames"][("C_p901", "T00")]
    assert len(decoded) == len(source)
    for got, src in zip(decoded, source):
        assert np.array_equal(got, _block_means_16(src))


def test_verifier_tells_8px_and_16px_releases_apart(fake_source):
    root = fake_source["root"]
    cfg8, cfg16 = _config(fake_source), _config16(fake_source)
    build_release(cfg8)
    build_release(cfg16)
    assert any("not block-constant" in e for e in _verify(cfg8, block_size=16, keep_ratio=K16).errors)
    as_8px = _verify(cfg16, source_root=root, mapping_path=cfg16.mapping_path)  # 16 x 16 blocks are also 8 x 8-constant
    assert any(f"{C.RELEASE_VIDEO} differs from BFDT of the source video" in e for e in as_8px.errors)
    assert any("also block-constant at 16 px" in w for w in _verify(cfg16).warnings)  # even without --source
    assert not any("also block-constant" in w for w in _verify(cfg8).warnings)


def test_16px_release_reuses_the_8px_numbering_when_given_its_mapping(fake_source):
    cfg8, cfg16 = _config(fake_source), _config16(fake_source, seed=99)
    build_release(cfg8)
    cfg16.private_dir.mkdir()
    shutil.copy2(cfg8.mapping_path, cfg16.mapping_path)  # what the release procedure does
    report = build_release(cfg16)
    assert report["mapping_created"] is False
    assert _mapping(cfg16) == _mapping(cfg8)
    for name in C.METADATA_FILES:
        assert (cfg16.release_root / "metadata" / name).read_bytes() == (cfg8.release_root / "metadata" / name).read_bytes()


def test_settings_that_are_not_dc_only_are_refused_before_any_write(fake_source):
    with pytest.raises(ValueError):
        build_release(_config(fake_source, block_size=16))  # k_r = 0.02 keeps 5 coefficients at block size 16
    assert not (fake_source["workdir"] / "release_out").exists()


def test_block_size_must_divide_the_frame_size(fake_source):
    with pytest.raises(ValueError):
        _config16(fake_source, expected_frame_size=(40, FRAME_H))  # 40 % 16 != 0


def test_verifier_refuses_settings_that_are_not_dc_only(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    with pytest.raises(ValueError):
        _verify(cfg, block_size=16, keep_ratio=0.02)


# --- one release per output/private folder pair --------------------------------------

def test_other_block_size_with_an_existing_private_dir_is_refused(fake_source):
    cfg8 = _config(fake_source)
    build_release(cfg8)
    before = tree_fingerprint(cfg8.private_dir), tree_fingerprint(cfg8.output_dir)
    with pytest.raises(ReleaseConflictError):
        build_release(_config16(fake_source, private_dir=cfg8.private_dir))
    assert (tree_fingerprint(cfg8.private_dir), tree_fingerprint(cfg8.output_dir)) == before
    assert not (fake_source["workdir"] / "out16").exists()


def test_16px_build_into_the_8px_release_folder_is_refused(fake_source):
    cfg8 = _config(fake_source)
    build_release(cfg8)
    before = tree_fingerprint(cfg8.output_dir), tree_fingerprint(cfg8.private_dir)
    with pytest.raises(ReleaseConflictError):  # a private dir that never built this release
        build_release(_config16(fake_source, output_dir=cfg8.output_dir))
    with pytest.raises(ReleaseConflictError):  # dry runs check too
        build_release(_config16(fake_source, output_dir=cfg8.output_dir, dry_run=True))
    with pytest.raises(ReleaseConflictError):  # the release's own private dir, other block size
        build_release(_config16(fake_source, output_dir=cfg8.output_dir, private_dir=cfg8.private_dir))
    assert (tree_fingerprint(cfg8.output_dir), tree_fingerprint(cfg8.private_dir)) == before
    assert not (fake_source["workdir"] / "private16").exists()


def test_private_dir_of_another_release_folder_is_refused(fake_source):
    cfg16 = _config16(fake_source)
    build_release(cfg16)
    elsewhere = _config16(fake_source, output_dir=fake_source["workdir"] / "elsewhere")
    with pytest.raises(ReleaseConflictError):
        build_release(elsewhere)
    assert not elsewhere.output_dir.exists()
    assert build_release(cfg16)["skipped"] == 4  # the release's own folders still resume


def _make_legacy(cfg, every=1):
    """Make folders look like ones built before claims existed: no claim, no marker, and no
    release_root in every `every`-th session record (the block-16 private dir has a mix)."""
    cfg.claim_path.unlink()
    cfg.marker_path.unlink()
    for i, path in enumerate(sorted(cfg.records_dir.glob("*.json"))):
        if i % every == 0:
            record = json.loads(path.read_text())
            del record["release_root"]
            path.write_text(json.dumps(record))


def test_build_claims_its_folders_with_one_token(fake_source):
    cfg = _config16(fake_source)
    build_release(cfg)
    claim, marker = json.loads(cfg.claim_path.read_text()), json.loads(cfg.marker_path.read_text())
    assert (claim["release_root"], claim["block_size"], claim["keep_ratio"]) == (str(cfg.release_root.resolve()), 16, K16)
    assert marker == {"token": claim["token"]} and len(claim["token"]) == 32
    assert cfg.marker_path.parent == cfg.output_dir  # beside KR-RPPG/, not inside the published tree
    assert stat.S_IMODE(os.stat(cfg.claim_path).st_mode) == 0o600
    assert _verify(cfg, block_size=16, keep_ratio=K16).errors == []


def test_first_build_stopped_before_any_record_still_resumes(fake_source):
    cfg = _config16(fake_source)
    build_release(cfg)
    for path in cfg.records_dir.glob("*.json"):
        path.unlink()  # as if stopped after the videos landed but before any record was written
    report = build_release(cfg)
    assert (report["built"], report["failed"]) == (4, 0)


def test_first_build_stopped_between_claim_and_marker_still_resumes(fake_source):
    cfg = _config16(fake_source)
    build_release(cfg)
    cfg.marker_path.unlink()
    shutil.rmtree(cfg.release_root)
    shutil.rmtree(cfg.records_dir)
    assert build_release(cfg)["built"] == 4


def test_adopting_an_older_release_stopped_before_the_marker_still_resumes(fake_source):
    cfg8, cfg16 = _config(fake_source), _config16(fake_source)
    build_release(cfg8)
    build_release(cfg16)
    _make_legacy(cfg8)
    build_release(cfg8)  # adopts the older release: writes claim, then marker
    cfg8.marker_path.unlink()  # as if stopped between the two writes
    assert build_release(cfg8)["skipped"] == 4
    assert cfg8.marker_path.is_file()
    # without a marker, a claimed private dir still needs its own files at the claimed path
    shutil.rmtree(cfg16.output_dir)
    cfg8.output_dir.rename(cfg16.output_dir)
    cfg16.marker_path.unlink()
    before = tree_fingerprint(cfg16.output_dir)
    with pytest.raises(ReleaseConflictError):
        build_release(cfg16)
    assert tree_fingerprint(cfg16.output_dir) == before


def test_renamed_release_folders_are_not_mixed_up(fake_source):
    work = fake_source["workdir"]
    cfg8, cfg16 = _config(fake_source), _config16(fake_source)
    build_release(cfg8)
    build_release(cfg16)
    cfg8.output_dir.rename(work / "moved_8px")
    cfg16.output_dir.rename(cfg8.output_dir)  # the block-16 release now sits where the block-8 command points
    before = tree_fingerprint(cfg8.output_dir)
    with pytest.raises(ReleaseConflictError):
        build_release(cfg8)
    assert tree_fingerprint(cfg8.output_dir) == before
    assert build_release(_config(fake_source, output_dir=work / "moved_8px"))["skipped"] == 4  # follows the move


def test_legacy_private_dir_still_resumes_its_own_release(fake_source):
    cfg8 = _config(fake_source)
    build_release(cfg8)
    _make_legacy(cfg8)
    assert build_release(cfg8)["skipped"] == 4
    assert json.loads(cfg8.marker_path.read_text())["token"] == json.loads(cfg8.claim_path.read_text())["token"]
    with pytest.raises(ReleaseConflictError):
        build_release(_config16(fake_source, private_dir=cfg8.private_dir))


def test_legacy_private_dir_cannot_build_into_another_release(fake_source):
    cfg8, cfg16 = _config(fake_source), _config16(fake_source)
    build_release(cfg8)
    build_release(cfg16)
    _make_legacy(cfg8)
    _make_legacy(cfg16)
    before = tree_fingerprint(cfg16.output_dir), tree_fingerprint(cfg8.private_dir)
    with pytest.raises(ReleaseConflictError):  # block-8 resume with --output-dir mistyped as the block-16 folder
        build_release(_config(fake_source, output_dir=cfg16.output_dir))
    assert (tree_fingerprint(cfg16.output_dir), tree_fingerprint(cfg8.private_dir)) == before


@pytest.mark.parametrize("target", ["typo", "release_tree", "nested"])
def test_legacy_private_dir_refuses_a_mistyped_output_dir(fake_source, target):
    cfg8 = _config(fake_source)
    build_release(cfg8)
    _make_legacy(cfg8)
    before = tree_fingerprint(cfg8.private_dir), tree_fingerprint(cfg8.output_dir)
    output = {"typo": fake_source["workdir"] / "release_otu", "release_tree": cfg8.release_root,
              "nested": cfg8.release_root / "p001"}[target]
    with pytest.raises(ReleaseConflictError):
        build_release(_config(fake_source, output_dir=output))
    assert (tree_fingerprint(cfg8.private_dir), tree_fingerprint(cfg8.output_dir)) == before
    assert not (fake_source["workdir"] / "release_otu").exists()


def test_mixed_legacy_records_still_pin_the_private_dir(fake_source):
    cfg16 = _config16(fake_source)
    build_release(cfg16)
    _make_legacy(cfg16, every=2)  # a mix: some records with release_root, some without
    before = tree_fingerprint(cfg16.private_dir)
    with pytest.raises(ReleaseConflictError):  # empty other folder
        build_release(_config16(fake_source, output_dir=fake_source["workdir"] / "empty"))
    with pytest.raises(ReleaseConflictError):  # other block size
        build_release(_config(fake_source, output_dir=fake_source["workdir"] / "empty", private_dir=cfg16.private_dir))
    assert tree_fingerprint(cfg16.private_dir) == before
    assert build_release(cfg16)["skipped"] == 4


def test_protected_folders_cannot_be_written(fake_source):
    root, work = fake_source["root"], fake_source["workdir"]
    common = ["build", "--source", str(root), "--crf-ild", str(fake_source["ild_xlsx"]),
              "--frame-size", f"{FRAME_W}x{FRAME_H}", "--workers", "1", "--seed", "1"]
    other_release = work / "other_release"
    other_release.mkdir()
    with pytest.raises(UnsafePathError):
        main([*common, "--output-dir", str(other_release), "--private-dir", str(work / "p"),
              "--protect", str(other_release)])
    with pytest.raises(UnsafePathError):
        main([*common, "--output-dir", str(work / "o"), "--private-dir", str(work / "p"), "--protect", str(work)])
    for flag in ([], ["--metadata-only"]):
        with pytest.raises(UnsafePathError):
            main([*common, *flag, "--output-dir", str(work / "o"), "--private-dir", str(work / "p"),
                  "--protect", str(work / "o")])
    assert not (work / "o").exists() and not (work / "p").exists()
    assert os.listdir(other_release) == []


# --- metadata-only refresh ------------------------------------------------------------

def _as_older_release(cfg):
    """Give a built release the 1.1 six-column participants.csv and dictionary (synthetic stand-ins)."""
    meta = cfg.release_root / "metadata"
    (meta / "participants.csv").write_text("participant_key,sex,age_group,glasses,makeup,accessories\n"
                                           "p001,F,60-69,0,1,0\np002,M,<60,1,0,1\np003,F,>=70,0,0,0\n")
    (meta / "participants_dictionary.csv").write_text("variable,variable_group,description,unit_or_coding,"
                                                      "availability_n_of_3\nparticipant_key,Identifier,x,-,3\n")


def _outside_metadata(cfg):
    """Independent of the builder: files (sha256, size, mtime, ctime) and folders (mtime, inode) outside metadata/."""
    meta = str(cfg.release_root / "metadata")
    files = {k: v for k, v in tree_fingerprint(cfg.output_dir).items() if not k.startswith("KR-rPPG/metadata/")}
    folders = {d: (os.stat(d).st_mtime_ns, os.stat(d).st_ino) for d, _dirs, _files in os.walk(cfg.output_dir)
               if d != meta}
    return files, folders


def _inodes(cfg):
    return {name: os.stat(cfg.release_root / "metadata" / name).st_ino for name in C.METADATA_FILES}


def test_metadata_refresh_rewrites_only_the_changed_metadata_files(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    expected = {name: (cfg.release_root / "metadata" / name).read_bytes() for name in C.METADATA_FILES}
    _as_older_release(cfg)
    assert _verify(cfg).errors  # the 6-column file fails the 24-column schema
    outside, inodes = _outside_metadata(cfg), _inodes(cfg)
    private = tree_fingerprint(cfg.private_dir)
    source = tree_fingerprint(fake_source["root"]), fake_source["ild_xlsx"].read_bytes()

    report = refresh_metadata(cfg)

    assert report["metadata_written"] == ["participants.csv", "participants_dictionary.csv"]
    assert report["metadata_unchanged"] == ["fold_assignment.csv", "task_session_availability.csv"]
    assert report["release_unchanged_outside_metadata"] is True and report["release_changes_outside_metadata"] == []
    assert report["source_unchanged"] is True and report["builder_version"] == __version__ == "1.5.0"
    assert report["release_entries_watched"] > 20
    assert {name: (cfg.release_root / "metadata" / name).read_bytes() for name in C.METADATA_FILES} == expected
    assert _outside_metadata(cfg) == outside  # every session file: same bytes, size, mtime and ctime
    after = _inodes(cfg)
    assert after["fold_assignment.csv"] == inodes["fold_assignment.csv"]
    assert after["task_session_availability.csv"] == inodes["task_session_availability.csv"]
    changed_private = {k for k, v in tree_fingerprint(cfg.private_dir).items() if private.get(k) != v}
    assert changed_private == {"metadata_report.json"}  # no claim, record or mapping is touched
    assert (tree_fingerprint(fake_source["root"]), fake_source["ild_xlsx"].read_bytes()) == source
    assert stat.S_IMODE(os.stat(cfg.metadata_report_path).st_mode) == 0o600
    assert _verify(cfg, source_root=fake_source["root"], mapping_path=cfg.mapping_path).errors == []
    assert refresh_metadata(cfg)["metadata_written"] == []  # a second run changes nothing


def test_metadata_refresh_dry_run_writes_nothing(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    _as_older_release(cfg)
    before = tree_fingerprint(cfg.output_dir), tree_fingerprint(cfg.private_dir)
    report = refresh_metadata(_config(fake_source, dry_run=True))
    assert report["would_write"] == ["participants.csv", "participants_dictionary.csv"]
    assert (tree_fingerprint(cfg.output_dir), tree_fingerprint(cfg.private_dir)) == before


def test_metadata_refresh_of_a_legacy_release_writes_no_claim(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    _make_legacy(cfg)  # a private folder from before claims existed: no claim, no marker
    _as_older_release(cfg)
    report = refresh_metadata(cfg)
    assert report["release_unchanged_outside_metadata"] is True
    assert not cfg.claim_path.exists() and not cfg.marker_path.exists()
    assert build_release(cfg)["skipped"] == 4  # a later full build still adopts and resumes the release


def test_metadata_refresh_refuses_other_folders_before_writing(fake_source):
    cfg8, cfg16 = _config(fake_source), _config16(fake_source)
    build_release(cfg8)
    build_release(cfg16)
    _as_older_release(cfg8)
    before = tree_fingerprint(cfg8.output_dir), tree_fingerprint(cfg16.private_dir)
    with pytest.raises(ReleaseConflictError):  # the block-16 private folder pointed at the block-8 release
        refresh_metadata(_config16(fake_source, output_dir=cfg8.output_dir))
    with pytest.raises(ReleaseConflictError):  # the block-8 private folder with the block-16 setting
        refresh_metadata(_config16(fake_source, output_dir=cfg8.output_dir, private_dir=cfg8.private_dir))
    empty = _config(fake_source, output_dir=fake_source["workdir"] / "empty_out",
                    private_dir=fake_source["workdir"] / "empty_private")
    with pytest.raises(ValueError):  # nothing built there: a refresh never starts a release
        refresh_metadata(empty)
    with pytest.raises(ValueError):
        refresh_metadata(_config(fake_source, participants=("p001",)))
    assert (tree_fingerprint(cfg8.output_dir), tree_fingerprint(cfg16.private_dir)) == before
    assert not empty.output_dir.exists() and not empty.private_dir.exists()


def test_metadata_refresh_never_draws_a_new_mapping(fake_source):
    cfg = _config(fake_source)
    build_release(cfg)
    cfg.mapping_path.rename(cfg.private_dir / "moved_map.csv")
    before = tree_fingerprint(cfg.output_dir)
    with pytest.raises(ValueError):
        refresh_metadata(cfg)
    assert not cfg.mapping_path.exists() and tree_fingerprint(cfg.output_dir) == before


def test_cli_metadata_only(fake_source, capsys):
    root, work = fake_source["root"], fake_source["workdir"]
    common = ["--source", str(root), "--crf-ild", str(fake_source["ild_xlsx"]), "--frame-size", f"{FRAME_W}x{FRAME_H}",
              "--output-dir", str(work / "o"), "--private-dir", str(work / "p")]
    assert main(["build", *common, "--seed", "1", "--workers", "1"]) == 0
    cfg = _config(fake_source, output_dir=work / "o", private_dir=work / "p")
    _as_older_release(cfg)
    capsys.readouterr()
    assert main(["build", *common, "--metadata-only", "--dry-run"]) == 0
    assert '"participants.csv"' in capsys.readouterr().out
    assert main(["build", *common, "--metadata-only"]) == 0
    assert '"release_unchanged_outside_metadata": true' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["build", *common, "--metadata-only", "--participants", "p001"])
    assert main(["verify", "--release-root", str(cfg.release_root), "--source", str(root), "--mapping",
                 str(cfg.mapping_path), "--crf-ild", str(fake_source["ild_xlsx"]), "--frame-size",
                 f"{FRAME_W}x{FRAME_H}"]) == 0


# --- CLI ------------------------------------------------------------------------

def test_cli_build_then_verify(fake_source):
    root = fake_source["root"]
    work = fake_source["workdir"]
    before = tree_fingerprint(root)
    common = ["--frame-size", f"{FRAME_W}x{FRAME_H}"]
    rc = main(
        ["build", "--source", str(root), "--output-dir", str(work / "o"), "--private-dir", str(work / "p"),
         "--crf-ild", str(fake_source["ild_xlsx"]), "--seed", "1", "--workers", "1", *common]
    )
    assert rc == 0
    rc = main(
        ["verify", "--release-root", str(work / "o" / "KR-rPPG"), "--source", str(root),
         "--mapping", str(work / "p" / "participant_id_map.csv"), "--crf-ild", str(fake_source["ild_xlsx"]), *common]
    )
    assert rc == 0
    assert tree_fingerprint(root) == before


def test_cli_builds_and_verifies_a_16px_release(fake_source, capsys):
    root = fake_source["root"]
    work = fake_source["workdir"]
    common = ["--frame-size", f"{FRAME_W}x{FRAME_H}", "--crf-ild", str(fake_source["ild_xlsx"])]
    rc = main(["build", "--source", str(root), "--output-dir", str(work / "o16"), "--private-dir", str(work / "p16"),
               "--seed", "1", "--workers", "1", "--block-size", "16", *common])
    assert rc == 0
    assert '"mapping_created": true' in capsys.readouterr().out
    record = json.loads(next((work / "p16" / "session_records").glob("*.json")).read_text())
    assert (record["settings"]["block_size"], record["settings"]["keep_ratio"]) == (16, 1 / 256)
    verify = ["verify", "--release-root", str(work / "o16" / "KR-rPPG"), "--source", str(root),
              "--mapping", str(work / "p16" / "participant_id_map.csv"), *common]
    assert main([*verify, "--block-size", "16"]) == 0
    assert main(verify) == 1  # the default 8 px render of the source does not match


@pytest.mark.parametrize("command", ["build", "verify"])
@pytest.mark.parametrize("block", ["1", "4", "7", "32"])
def test_cli_rejects_unsupported_block_sizes(tmp_path, capsys, command, block):
    paths = (["--output-dir", str(tmp_path / "o"), "--private-dir", str(tmp_path / "p"), "--dry-run"]
             if command == "build" else ["--release-root", str(tmp_path / "KR-rPPG")])
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--source", str(tmp_path / "src"), *paths, "--block-size", block])
    assert exit_info.value.code == 2
    assert "--block-size" in capsys.readouterr().err
    assert os.listdir(tmp_path) == []


def test_cli_returns_error_code_when_sessions_fail(fake_source):
    root = fake_source["root"]
    work = fake_source["workdir"]
    (root / "C" / "02-groupb" / "p901" / "T00" / C.HR_SYNC).write_text("frame,timestamp,hr\n")
    rc = main(
        ["build", "--source", str(root), "--output-dir", str(work / "o"), "--private-dir", str(work / "p"),
         "--crf-ild", str(fake_source["ild_xlsx"]), "--workers", "1", "--frame-size", f"{FRAME_W}x{FRAME_H}"]
    )
    assert rc == 1

import csv
import io
import shutil
from decimal import Decimal

import pytest

from conftest import EPOCH_T0
from krrppg_release import constants as C
from krrppg_release.session_files import (
    SessionDataError,
    frame_sequence_warnings,
    load_session_tables,
    relative_timestamps,
    render_tables,
    validate_tables,
)


def _session_dir(fake_source):
    return fake_source["root"] / "incheon" / "02-groupb" / "p901" / "T00"


def _parse(data):
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    return rows[0], rows[1:]


def _timestamps(rendered, name):
    _header, rows = _parse(rendered[name])
    return [row[C.TIMESTAMP_COLUMN[name]] for row in rows]


def test_relative_timestamps_are_exact_decimal_differences():
    values = [EPOCH_T0, "1000000000.0333333", "1000000000.6666667"]  # synthetic clock (see conftest)
    assert relative_timestamps(values) == ["0.0000000", "0.0333333", "0.6666667"]


@pytest.mark.parametrize("bad", ["abc", "1e9", "", "-1.5", "nan"])
def test_relative_timestamps_reject_non_plain_numbers(bad):
    with pytest.raises(SessionDataError):
        relative_timestamps([EPOCH_T0, bad])


def test_float_round_trip_differences_between_files_are_accepted(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    assert validate_tables(tables) == []


def test_signal_row_count_mismatch_is_fatal(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    tables[C.BVP_SYNC] = tables[C.BVP_SYNC][:-1]
    with pytest.raises(SessionDataError):
        validate_tables(tables)


def test_face_bbox_row_count_mismatch_is_a_warning(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    tables[C.FACE_BBOXES] = tables[C.FACE_BBOXES][:-1]
    warnings = validate_tables(tables)
    assert len(warnings) == 1
    assert C.FACE_BBOXES in warnings[0]


def test_timestamp_disagreement_beyond_tolerance_is_fatal(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    frame, ts, hr = tables[C.HR_SYNC][2]
    tables[C.HR_SYNC][2] = [frame, str(Decimal(ts) + Decimal("0.01")), hr]
    with pytest.raises(SessionDataError):
        validate_tables(tables)


def test_frame_column_disagreement_is_fatal(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    _frame, ts, spo2 = tables[C.SPO2_SYNC][1]
    tables[C.SPO2_SYNC][1] = ["99", ts, spo2]
    with pytest.raises(SessionDataError):
        validate_tables(tables)


def test_ragged_rows_are_fatal(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    tables[C.BVP_SYNC][0] = tables[C.BVP_SYNC][0] + ["extra"]
    with pytest.raises(SessionDataError):
        validate_tables(tables)


def test_unexpected_header_is_rejected(fake_source, tmp_path):
    copy = tmp_path / "session_copy"
    shutil.copytree(_session_dir(fake_source), copy)
    (copy / C.FRAMES_TIMESTAMP).write_text("frame,time\n0,1.0\n")
    with pytest.raises(SessionDataError):
        load_session_tables(copy)


def test_relative_mode_shares_one_timestamp_sequence_starting_at_zero(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    rendered = render_tables(tables, "relative")
    sequences = [_timestamps(rendered, name) for name in C.TIMESTAMP_COLUMN]
    assert all(seq == sequences[0] for seq in sequences)
    assert sequences[0][0] == "0.0000000"
    assert all(Decimal(v) < 86400 for v in sequences[0])


def test_value_columns_are_copied_verbatim(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    rendered = render_tables(tables, "relative")
    _h, bvp = _parse(rendered[C.BVP_SYNC])
    assert [row[1] for row in bvp] == [row[1] for row in tables[C.BVP_SYNC]]
    _h, hr = _parse(rendered[C.HR_SYNC])
    assert [(row[0], row[2]) for row in hr] == [(row[0], row[2]) for row in tables[C.HR_SYNC]]
    header, _rows = _parse(rendered[C.SPO2_SYNC])
    assert tuple(header) == C.TABLE_HEADERS[C.SPO2_SYNC]


def test_absolute_mode_uses_frame_timestamps_everywhere(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    rendered = render_tables(tables, "absolute")
    source = [row[1] for row in tables[C.FRAMES_TIMESTAMP]]
    for name in C.TIMESTAMP_COLUMN:
        assert _timestamps(rendered, name) == source


def test_unknown_timestamp_mode_is_rejected(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    with pytest.raises(ValueError):
        render_tables(tables, "shifted")


# --- capture-sequence warnings ------------------------------------------------------

def test_frame_id_reset_and_long_gap_are_reported():
    frame_ids = ["0", "1", "2", "0", "1"]
    stamps = [Decimal(v) for v in ("0.0", "0.033", "0.066", "5.2", "5.233")]
    warnings = frame_sequence_warnings(frame_ids, stamps)
    assert any("goes backwards 1 time" in w and "2 duplicate" in w for w in warnings)
    assert any("1 inter-frame gap" in w for w in warnings)


def test_regular_sequence_with_dropped_frames_is_not_reported():
    frame_ids = ["0", "1", "3", "4"]
    stamps = [Decimal(v) for v in ("0.0", "0.033", "0.1", "0.133")]
    assert frame_sequence_warnings(frame_ids, stamps) == []


def test_validate_reports_sequence_warnings(fake_source):
    tables = load_session_tables(_session_dir(fake_source))
    for name in (C.FRAMES_TIMESTAMP, C.HR_SYNC, C.SPO2_SYNC):
        tables[name][3][0] = "0"
    warnings = validate_tables(tables)
    assert any("goes backwards" in w for w in warnings)

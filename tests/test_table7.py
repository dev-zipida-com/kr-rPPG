"""Table 7 analysis on synthetic signals and tables (no real data): reference HR, row selection, statistics."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.signal import butter, filtfilt

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("table7_hr_distributions",
                                               REPO / "analysis" / "table7" / "table7_hr_distributions.py")
table7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(table7)


def _sine(freq_hz, fs, n, amplitude=1.0, phase=0.3):
    return amplitude * np.sin(2 * np.pi * freq_hz * np.arange(n) / fs + phase)


# ----------------------------------------------------------------------------- reference heart rate
@pytest.mark.parametrize("n, expected", [(1, 1), (2, 2), (3, 4), (1800, 2048), (2048, 2048), (2049, 4096)])
def test_next_pow2(n, expected):
    assert table7.next_pow2(n) == expected


def test_next_pow2_rejects_empty_signal():
    with pytest.raises(ValueError):
        table7.next_pow2(0)


def test_reference_hr_of_a_sinusoid_on_the_frequency_grid():
    # fs = 32 Hz, 2048 samples -> nfft 2048, bins of 1/64 Hz; 1.25 Hz = bin 80
    assert table7.reference_hr(_sine(1.25, 32.0, 2048), 32.0) == pytest.approx(75.0)


def test_reference_hr_zero_pads_to_the_next_power_of_two():
    # 1500 samples at 30 Hz: unpadded bins of 0.02 Hz would give exactly 66 bpm for 1.1 Hz; the recipe pads to
    # 2048 samples, whose nearest bin is 75 * 30 / 2048 Hz
    hr = table7.reference_hr(_sine(1.1, 30.0, 1500), 30.0)
    assert hr == pytest.approx(75 * 30.0 / 2048 * 60)
    assert hr != pytest.approx(66.0)


@pytest.mark.parametrize("freq_hz, bpm", [(0.75, 45.0), (2.5, 150.0)])
def test_reference_hr_band_edges_are_inclusive(freq_hz, bpm):
    assert table7.reference_hr(_sine(freq_hz, 32.0, 2048), 32.0) == pytest.approx(bpm)


def test_reference_hr_ignores_strong_components_outside_the_band():
    fs, n = 32.0, 2048
    x = _sine(0.3, fs, n, 2.0) + _sine(3.5, fs, n, 2.0) + _sine(1.25, fs, n, 0.5)
    assert table7.reference_hr(x, fs) == pytest.approx(75.0)


def test_reference_hr_never_leaves_the_search_band():
    rng = np.random.default_rng(3)
    fs, n = 30.0, 1800
    for freq_hz in (0.2, 0.5, 3.0, 6.0):
        hr = table7.reference_hr(_sine(freq_hz, fs, n) + 0.01 * rng.standard_normal(n), fs)
        assert 45.0 <= hr <= 150.0


def _independent_hr(x, fs):
    """The documented recipe with numpy.fft: order-1 Butterworth 0.75-2.5 Hz, filtfilt, no detrending, padding."""
    b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype="bandpass")
    nfft = 1 << (len(x) - 1).bit_length()
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    power = np.abs(np.fft.rfft(filtfilt(b, a, x), n=nfft)) ** 2
    mask = (freqs >= 0.75) & (freqs <= 2.5)
    return 60 * freqs[mask][np.argmax(power[mask])]


@pytest.mark.parametrize("fs, n", [(30.0, 1800), (64.0, 3000), (59.3, 3900)])
def test_reference_hr_matches_an_independent_fft_implementation(fs, n):
    rng = np.random.default_rng(int(fs * n))
    for _ in range(40):  # noise-dominated signals with offset and drift: the peak depends on every recipe step
        x = 0.3 * _sine(rng.uniform(0.8, 2.4), fs, n) + rng.standard_normal(n) + 50 + np.linspace(0, 20, n)
        assert table7.reference_hr(x, fs) == pytest.approx(_independent_hr(x, fs))


@pytest.mark.parametrize("signal, fs", [([0.0] * 50 + [np.nan] + [0.0] * 50, 30.0),  # non-finite sample
                                        (np.ones(9), 30.0),                          # too short for filtfilt
                                        (np.ones(100), 4.0)])                        # 2.5 Hz above Nyquist
def test_reference_hr_rejects_unusable_input(signal, fs):
    with pytest.raises(ValueError):
        table7.reference_hr(signal, fs)


# ----------------------------------------------------------------------------- descriptive statistics
def test_describe_and_cells_use_ddof1_linear_percentiles_and_one_decimal():
    d = table7.describe(np.arange(1, 11))
    assert d[0] == 10
    assert d[2] == pytest.approx(np.sqrt(82.5 / 9))  # SD with ddof = 1
    assert (d[4], d[5]) == pytest.approx((3.25, 7.75))  # linear interpolation
    assert table7.cells(d) == ["10", "5.5", "3.0", "5.5", "3.2", "7.8", "1.0", "10.0"]


def test_format_row_and_matching_cells():
    kr = table7.EXPECTED["KR-rPPG"]
    row = table7.format_row(table7.cells(kr))
    assert "293" in row and "81.0 +/- 13.8" in row and "80.4 (70.8-90.1)" in row and "52.7-130.1" in row
    assert table7.matching_cells(kr, kr) == 8
    assert table7.matching_cells(kr[:2] + (13.9,) + kr[3:], kr) == 7
    assert table7.matching_cells((293, 81.04) + kr[2:], kr) == 8  # compared after rounding to one decimal


def test_expected_values_cover_the_four_rows_and_three_comparisons():
    assert list(table7.EXPECTED) == ["KR-rPPG", *table7.EXTERNAL]
    assert all(len(v) == 8 for v in table7.EXPECTED.values())
    assert set(table7.EXPECTED_P) == set(table7.EXTERNAL)


# ----------------------------------------------------------------------------- tests
def test_holm_adjustment_is_monotone_and_capped():
    assert table7.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert table7.holm([0.5, 0.6, 0.01]) == pytest.approx([1.0, 1.0, 0.03])


def test_compare_with_kr_welch_mann_whitney_and_holm():
    kr = np.array([1.0, 2.0, 3.0])
    others = {"MMPD": np.array([4.0, 5.0, 6.0]), "PURE": np.array([1.5, 2.5, 9.0, 12.0]),
              "UBFC-Phys": np.array([0.5, 2.2, 3.1])}
    results = table7.compare_with_kr(kr, others)
    assert list(results) == ["MMPD", "PURE", "UBFC-Phys"]
    # completely separated samples of 3 and 3: exact two-sided p = 2 / C(6, 3)
    assert results["MMPD"]["mw_p"] == pytest.approx(0.1)
    # Welch t-test from its definition
    a, b = kr, others["PURE"]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    assert results["PURE"]["welch_p"] == pytest.approx(2 * stats.t.sf(abs(t), df))
    for test in ("welch", "mw"):
        raw = [results[n][f"{test}_p"] for n in results]
        assert [results[n][f"{test}_holm"] for n in results] == pytest.approx(table7.holm(raw))


def _results(welch=(0.05998, 0.0625, 0.724), mw=(0.106, 0.00104, 0.602)):
    welch_holm, mw_holm = table7.holm(list(welch)), table7.holm(list(mw))
    return {n: {"welch_p": welch[i], "welch_holm": welch_holm[i], "mw_p": mw[i], "mw_holm": mw_holm[i]}
            for i, n in enumerate(table7.EXTERNAL)}


def test_manuscript_statements_hold_for_the_original_p_values():
    assert [ok for _text, ok in table7.manuscript_checks(_results())] == [True, True, True]


@pytest.mark.parametrize("welch, mw, failing", [
    ((0.0594, 0.0625, 0.724), (0.106, 0.00104, 0.602), 0),   # a Welch p below 0.060 at three decimals
    ((0.05998, 0.0625, 0.724), (0.106, 0.0016, 0.602), 1),   # PURE Mann-Whitney p = 0.002
    ((0.05998, 0.0625, 0.724), (0.01, 0.00104, 0.602), 2),   # MMPD significant after Holm
])
def test_manuscript_statements_detect_deviations(welch, mw, failing):
    flags = [ok for _text, ok in table7.manuscript_checks(_results(welch, mw))]
    assert flags[failing] is False


def test_sig3_keeps_three_significant_digits():
    assert table7.sig3(0.0599813) == "0.0600"
    assert table7.sig3(0.0010387) == "0.00104"
    assert table7.sig3(0.72391) == "0.724"


# ----------------------------------------------------------------------------- row selection (synthetic files)
def _write_pure(root, name, freq_hz, fs=64.0, n=4096):
    folder = root / name
    folder.mkdir(parents=True)
    t0 = 1_400_000_000_000_000_000
    packages = [{"Timestamp": int(t0 + round(i * 1e9 / fs)), "Value": {"waveform": round(60 + 40 * v)}}
                for i, v in enumerate(_sine(freq_hz, fs, n))]
    (folder / f"{name}.json").write_text(json.dumps({"/FullPackage": packages}))


def test_pure_uses_only_the_steady_session_of_each_subject(tmp_path):
    _write_pure(tmp_path, "01-01", 1.25)
    _write_pure(tmp_path, "01-02", 2.0)  # another task: ignored
    _write_pure(tmp_path, "02-01", 1.0)
    values = table7.pure_values(tmp_path)
    assert values == pytest.approx([75.0, 60.0], abs=0.01)


def _write_mmpd(root, subject, name, motion, freq_hz, n=1800):
    from scipy.io import savemat

    folder = root / f"subject{subject}"
    folder.mkdir(parents=True, exist_ok=True)
    savemat(str(folder / name), {"GT_ppg": _sine(freq_hz, 30.0, n)[None, :], "motion": motion})


def test_mmpd_selects_exactly_stationary_and_averages_per_subject(tmp_path):
    _write_mmpd(tmp_path, 1, "a_0.mat", "Stationary", 1.0)
    _write_mmpd(tmp_path, 1, "a_1.mat", "Stationary (after exercise)", 2.2)  # excluded
    _write_mmpd(tmp_path, 1, "a_2.mat", "Stationary", 1.25)
    _write_mmpd(tmp_path, 1, "a_3.mat", "Stationary", 2.0)
    _write_mmpd(tmp_path, 2, "b_0.mat", "Walking", 2.2)  # excluded
    _write_mmpd(tmp_path, 2, "b_1.mat", "Stationary", 1.0)
    values = table7.mmpd_values(tmp_path)
    bin_bpm = 30.0 / 2048 * 60
    assert len(values) == 2
    assert values == pytest.approx([(60.0 + 75.0 + 120.0) / 3, 60.0], abs=bin_bpm)  # mean (85), not median (75)


def _write_ubfc_phys(root, subject, task, freq_hz, n=64 * 60):
    folder = root / subject
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_sine(freq_hz, 64.0, n)).to_csv(folder / f"bvp_{subject}_{task}.csv", header=False, index=False)


def test_ubfc_phys_uses_the_resting_task_of_every_subject(tmp_path, capsys):
    _write_ubfc_phys(tmp_path, "s10", "T1", 1.0)
    _write_ubfc_phys(tmp_path, "s2", "T1", 1.25)
    _write_ubfc_phys(tmp_path, "s2", "T2", 2.0)  # other task: ignored
    _write_ubfc_phys(tmp_path, "s3", "T3", 2.0)  # no T1: skipped and reported
    (tmp_path / "s4.zip").write_bytes(b"")  # not a subject folder
    values = table7.ubfc_phys_values(tmp_path)
    assert values == pytest.approx([75.0, 60.0], abs=64.0 / 4096 * 60)  # numeric subject order: s2, s10
    assert "1 subject folder(s) without a T1 BVP file" in capsys.readouterr().err


def _kr_table(path, rows):
    pd.DataFrame(rows, columns=["model", "fold", "participant_key", "task", "label_hr"]).to_csv(path, index=False)


def test_kr_uses_the_deepphys_label_of_the_stationary_task(tmp_path):
    rows = []
    for i, label in enumerate((70.0, 80.0, np.nan)):
        key = f"p{901 + i}"
        rows += [("deepphys", str(i), key, "T01", label), ("deepphys", str(i), key, "T00", 120.0),
                 ("physnet", str(i), key, "T01", 50.0)]
    _kr_table(tmp_path / "kr.csv", rows)
    assert sorted(table7.kr_values(tmp_path / "kr.csv")) == [70.0, 80.0]  # missing label ignored


def test_kr_refuses_duplicate_rows(tmp_path):
    _kr_table(tmp_path / "kr.csv", [("deepphys", "0", "p901", "T01", 70.0), ("deepphys", "1", "p901", "T01", 71.0)])
    with pytest.raises(SystemExit):
        table7.kr_values(tmp_path / "kr.csv")


def test_main_prints_aggregates_and_flags_mismatches(tmp_path, capsys, monkeypatch):
    for variable in ("KR_RPPG_WORK", "PURE_ROOT", "MMPD_ROOT", "UBFC_PHYS_ROOT"):
        monkeypatch.delenv(variable, raising=False)
    _kr_table(tmp_path / "kr.csv", [("deepphys", "0", f"p{900 + i}", "T01", 60.0 + i) for i in range(1, 21)])
    _write_pure(tmp_path / "pure", "01-01", 1.25)
    _write_pure(tmp_path / "pure", "02-01", 1.0)
    assert table7.main(["--kr", str(tmp_path / "kr.csv"), "--pure", str(tmp_path / "pure")]) == 0
    out = capsys.readouterr().out
    assert "KR-rPPG" in out and "PURE" in out and "MMPD " not in out
    assert "MISMATCH" in out and "need all three public datasets" in out


def test_main_needs_an_input(monkeypatch):
    for variable in ("KR_RPPG_WORK", "PURE_ROOT", "MMPD_ROOT", "UBFC_PHYS_ROOT"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(SystemExit):
        table7.main([])

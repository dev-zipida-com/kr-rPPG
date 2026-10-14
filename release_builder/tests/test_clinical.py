import dataclasses
import math
import re
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

import numpy as np
import pandas as pd
import pytest

from conftest import CRF_COLUMNS, CRF_ROWS, CRF_SHEET, FEMALE, ILD_SHEET, _write_sheet, tree_fingerprint
from conftest import ckd_epi_2009 as reference_ckd_epi_2009
from krrppg_release.clinical import (
    CATEGORICAL_COLUMNS,
    CODINGS,
    DICTIONARY,
    FPG_MIN_PLAUSIBLE,
    NUMERIC_COLUMNS,
    PARTICIPANT_CATEGORIES,
    PARTICIPANT_COLUMNS,
    PLAUSIBLE_RANGES,
    SBP_MIN_PLAUSIBLE,
    Coding,
    CrfSources,
    apply_quality_rules,
    build_participant_table,
    ckd_epi_2009,
    derive_rows,
    load_raw,
    other_comorbidity_count,
    participants_dictionary,
    released_bmi,
)

MAPPING = {"incheon_p901": "p002", "incheon_p902": "p003", "seoul_p903": "p001"}
# Manuscript Table 6: names and order of the 24 participants.csv columns.
TABLE6 = (
    "participant_key", "site", "age", "sex", "height", "weight", "bmi", "makeup", "glasses", "accessories",
    "sbp", "dbp", "hr_rest", "hba1c", "fpg", "creatinine", "gfr", "ast", "alt",
    "dm", "htn", "dyslipidemia", "ild", "other_comorbidity_count",
)
GROUPS = {
    "Identifier": ["participant_key"],
    "Demographics": ["site", "age", "sex", "height", "weight", "bmi"],
    "Acquisition appearance indicators": ["makeup", "glasses", "accessories"],
    "Clinical measurements": ["sbp", "dbp", "hr_rest", "hba1c", "fpg", "creatinine", "gfr", "ast", "alt"],
    "Comorbidity indicators": ["dm", "htn", "dyslipidemia", "ild", "other_comorbidity_count"],
}
# 1.4.0: every column is coded as manuscript Table 6 says. Text and 0/1 columns keep their values ...
TABLE6_CATEGORIES = {
    "site": ("incheon", "saintvincent", "seoul"), "sex": ("M", "F"),
    "makeup": ("0", "1"), "glasses": ("0", "1"), "accessories": ("0", "1"),
    "dm": ("0", "1"), "htn": ("0", "1"), "dyslipidemia": ("0", "1"), "ild": ("0", "1"),
}
# ... and the other columns are numbers: column -> (source, kind, step, decimals). age and the comorbidity count are
# whole numbers; the others are rounded to a step (bmi from the rounded height and weight, released_bmi).
TABLE6_NUMBERS = {
    "age": ("age", "integer", "1", 0), "height": ("height", "round", "5", 0), "weight": ("weight", "round", "2", 0),
    "bmi": ("bmi", "round", "0.1", 1),
    "sbp": ("sbp", "round", "5", 0), "dbp": ("dbp", "round", "5", 0), "hr_rest": ("hr_rest", "round", "2", 0),
    "hba1c": ("hba1c", "round", "0.5", 1), "fpg": ("fpg", "round", "10", 0),
    "creatinine": ("creatinine", "round", "0.1", 1), "gfr": ("egfr", "round", "5", 0),
    "ast": ("ast", "round", "5", 0), "alt": ("alt", "round", "5", 0),
    "other_comorbidity_count": ("other_comorbidity_count", "integer", "1", 0),
}
# participants_dictionary.csv: description and unit_or_coding of manuscript Table 6 (spelled 'creatinine'), plus the
# notes of the two kept corrections that add information (gfr, renal disease) and the 1.2.1 hr_rest clarification;
# '{observed}' is the observed range of the count.
TABLE6_TEXT = {
    "participant_key": ("Pseudonymized participant identifier (pNNN; e.g., p001)", "-"),
    "site": ("Collection site (seoul / incheon / saintvincent)", "-"),
    "age": ("Age at enrollment", "years"),
    "sex": ("Biological sex", "M/F"),
    "height": ("Body height", "cm, rounded to 5"),
    "weight": ("Body weight", "kg, rounded to 2"),
    "bmi": ("Body mass index from rounded height/weight", "kg/m^2, 1 decimal"),
    "makeup": ("Facial makeup present at recording", "0/1"),
    "glasses": ("Eyeglasses worn at recording", "0/1"),
    "accessories": ("Facial accessories worn at recording", "0/1"),
    "sbp": ("Systolic blood pressure", "mmHg, rounded to 5"),
    "dbp": ("Diastolic blood pressure", "mmHg, rounded to 5"),
    "hr_rest": (("Resting heart rate (CRF vital sign; not the benchmark reference HR, which is derived from the "
                 "contact pulse waveform in each session's BVP_synchronized.csv, nor the CMS50E device pulse rate in "
                 "HR_synchronized.csv)"), "bpm, rounded to 2"),
    "hba1c": ("Glycated hemoglobin", "%, rounded to 0.5"),
    "fpg": ("Fasting plasma glucose", "mg/dL, rounded to 10"),
    "creatinine": ("Serum creatinine", "mg/dL, rounded to 0.1"),
    "gfr": (("eGFR (CKD-EPI); computed from serum creatinine with the CKD-EPI 2009 equation (no race coefficient) "
             "when the CRF value is missing"), "mL/min/1.73 m^2, rounded to 5"),
    "ast": ("Aspartate aminotransferase", "U/L, rounded to 5"),
    "alt": ("Alanine aminotransferase", "U/L, rounded to 5"),
    "dm": ("Diabetes mellitus", "0/1"),
    "htn": ("Hypertension", "0/1"),
    "dyslipidemia": ("Dyslipidemia", "0/1"),
    "ild": ("Interstitial lung disease (incheon site only; blank = not ascertained)", "0/1"),
    "other_comorbidity_count": (("Number of positive indicators among cardiovascular disease, cerebrovascular "
                                 "disease, cancer, renal disease, and lung disease; renal disease is also counted "
                                 "when eGFR is below 15 mL/min/1.73 m^2"), "0-5 possible{observed}"),
}
# The former blanking rules are only counted ('..._flagged_unchanged'); the two corrections that add information
# (eGFR computed from creatinine, eGFR < 15 counted as renal disease) are kept.
QUALITY_COUNT_KEYS = [
    "height_outside_120_210_flagged_unchanged", "bmi_outside_13_60_flagged_unchanged",
    "sbp_below_70_flagged_unchanged", "dbp_not_below_sbp_flagged_unchanged",
    "egfr_creatinine_discordant_flagged_unchanged", "egfr_computed_from_creatinine",
    "ast_isolated_extreme_flagged_unchanged", "renal_disease_recoded_from_egfr_below_15",
    "fpg_below_40_flagged_unchanged", "hr_outside_30_200_flagged_unchanged",
]
# Phrases of the 1.3.0 dictionary that described values removed by a rule, and of the former categories; none may
# remain.
RULE_WORDING = ("left blank", "blank for", "blank, together", "removed by", "data-quality", "entry error", "treated as",
                "outside 120-210", "outside 13-60", "below 70 mmhg", "below 40 mg/dl", "not below sbp", "disagree",
                "differ by", "isolated extreme")
CATEGORY_WORDING = ("band", "bins", "class", "cut-point", "top-coded", ">=", "<", "not released", "even multiple")

# The synthetic CRF of conftest.py (seeded generator), keyed by screening number, and the CRF item behind each
# numeric column; used to recompute the expected cells independently of the package.
SYNTHETIC_CRF = {row[0]: dict(zip(CRF_COLUMNS, row)) for row in CRF_ROWS}
SCREENING = {"incheon_p901": "04-S901", "incheon_p902": "04-S902", "seoul_p903": "01-S903"}
CRF_ITEM = {"height": "Height", "weight": "Weight", "sbp": "SBP", "dbp": "DBP", "hr_rest": "HR", "hba1c": "HbA1c",
            "fpg": "FPG", "creatinine": "Creatinine", "gfr": "GFR(CKD-EPI)", "ast": "AST", "alt": "ALT"}


def _crf(fake_source):
    return CrfSources.for_source(fake_source["root"], ild_xlsx=fake_source["ild_xlsx"])


def _by_id(rows):
    return {row[0]: dict(zip(PARTICIPANT_COLUMNS, row)) for row in rows}


def _is_numeric_cell(column, cell):
    """Plain decimal text with the column's decimals that is a multiple of its step."""
    coding = CODINGS[column]
    pattern = r"(0|[1-9][0-9]*)" + (rf"\.[0-9]{{{coding.decimals}}}" if coding.decimals else "")
    return re.fullmatch(pattern, cell) is not None and Decimal(cell) % Decimal(coding.step) == 0


def _rows_with(fake_source, **edits):
    """derive_rows after editing raw CRF values: edits = {key: {column: value}}."""
    raw = load_raw(_crf(fake_source), list(MAPPING))
    for key, values in edits.items():
        for column, value in values.items():
            raw.loc[key, column] = value
    data, counts = apply_quality_rules(raw)
    return _by_id(derive_rows(data, MAPPING)), counts


def _half_even(quotient, step, decimals):
    """Exact decimal half-to-even rounding of `quotient` steps; never called within 1e-6 of a half step."""
    assert abs(quotient - quotient.to_integral_value(ROUND_FLOOR) - Decimal("0.5")) > Decimal("1e-6")
    return f"{quotient.to_integral_value(ROUND_HALF_EVEN) * Decimal(step):.{decimals}f}"


def _reference_cell(value, step, decimals):
    """Exact decimal arithmetic, half to even (independent of the package's numpy rule).

    The synthetic values are never within 1e-6 of a half step (asserted), where binary and decimal rounding
    could disagree, so both rules must give the same cell.
    """
    return _half_even(Decimal(repr(float(value))) / Decimal(step), step, decimals)


def _reference_bmi(height_cell, weight_cell):
    """weight / (height / 100)^2 of the released cells in exact decimals, one decimal ('' when either is blank)."""
    if height_cell == "" or weight_cell == "":
        return ""
    return _half_even(Decimal(weight_cell) / (Decimal(height_cell) / 100) ** 2 / Decimal("0.1"), "0.1", 1)


def _expected_numbers(key):
    """age, height, weight, bmi and the nine measurement cells of one synthetic participant (no value is blanked)."""
    crf = SYNTHETIC_CRF[SCREENING[key]]
    cells = {"age": str(crf["Age"])}
    for column, (_source, _kind, step, decimals) in TABLE6_NUMBERS.items():
        if column not in CRF_ITEM:
            continue
        value = crf[CRF_ITEM[column]]
        if column == "gfr" and value == "-":  # missing CRF eGFR: CKD-EPI 2009 from creatinine, age and sex
            value = reference_ckd_epi_2009(crf["Creatinine"], crf["Age"], crf["Sex"] == FEMALE)
        cells[column] = "" if value in ("-", "ND") else _reference_cell(value, step, decimals)
    cells["bmi"] = _reference_bmi(cells["height"], cells["weight"])
    return cells


def test_columns_are_the_24_table6_variables_in_order():
    assert PARTICIPANT_COLUMNS == TABLE6 and len(PARTICIPANT_COLUMNS) == 24
    assert set(CODINGS) == set(PARTICIPANT_COLUMNS[1:]) and list(CODINGS) == list(PARTICIPANT_COLUMNS[1:])
    assert list(DICTIONARY) == list(PARTICIPANT_COLUMNS)
    assert {column: DICTIONARY[column][0] for column in PARTICIPANT_COLUMNS} == {
        column: group for group, columns in GROUPS.items() for column in columns
    }


def test_every_column_is_coded_as_in_table6():
    assert CATEGORICAL_COLUMNS == tuple(TABLE6_CATEGORIES) and NUMERIC_COLUMNS == tuple(TABLE6_NUMBERS)
    assert set(CATEGORICAL_COLUMNS) | set(NUMERIC_COLUMNS) == set(PARTICIPANT_COLUMNS[1:])
    assert PARTICIPANT_CATEGORIES == TABLE6_CATEGORIES  # categories only for the text and 0/1 columns
    assert {c: (CODINGS[c].source, CODINGS[c].kind, CODINGS[c].step, CODINGS[c].decimals)
            for c in NUMERIC_COLUMNS} == TABLE6_NUMBERS
    assert all(CODINGS[c].categories == () for c in NUMERIC_COLUMNS)
    assert {CODINGS[c].kind for c in CATEGORICAL_COLUMNS} == {"text", "flag"}


def test_plausible_ranges_are_sanity_limits_on_each_columns_grid():
    # bmi has no range of its own: verify checks it against the row's height and weight (released_bmi)
    assert set(PLAUSIBLE_RANGES) == set(NUMERIC_COLUMNS) - {"bmi"}
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        assert _is_numeric_cell(column, low) and _is_numeric_cell(column, high) and Decimal(low) < Decimal(high)
    assert {c: PLAUSIBLE_RANGES[c] for c in ("age", "height", "weight", "other_comorbidity_count")} == {
        "age": ("0", "120"), "height": ("50", "250"), "weight": ("10", "300"), "other_comorbidity_count": ("0", "5")}
    # the released height outside 120-210 cm (1.4.0) must pass; the count has five indicators, no top-coding
    assert Decimal(PLAUSIBLE_RANGES["height"][0]) < 120 and PLAUSIBLE_RANGES["other_comorbidity_count"][1] == "5"
    # 1.4.0: the sbp and fpg lower bounds are not the former blanking thresholds
    assert PLAUSIBLE_RANGES["sbp"] == ("50", "300") and PLAUSIBLE_RANGES["fpg"] == ("10", "1000")
    assert Decimal(PLAUSIBLE_RANGES["sbp"][0]) < SBP_MIN_PLAUSIBLE
    assert Decimal(PLAUSIBLE_RANGES["fpg"][0]) < FPG_MIN_PLAUSIBLE
    unchanged = {"dbp": ("20", "200"), "hr_rest": ("20", "250"), "hba1c": ("3.0", "20.0"), "creatinine": ("0.1", "20.0"),
                 "gfr": ("0", "200"), "ast": ("0", "5000"), "alt": ("0", "5000")}
    assert {column: PLAUSIBLE_RANGES[column] for column in unchanged} == unchanged


def test_rows_follow_the_table6_coding(fake_source):
    rows = _by_id(build_participant_table(_crf(fake_source), MAPPING).rows)
    assert list(rows) == ["p001", "p002", "p003"]  # sorted by release ID
    # The synthetic CRF rows are drawn inside the ranges of conftest.CRF_SPECS; the numbers are recomputed from the
    # synthetic values (_expected_numbers), independently of the package's rounding code.
    # incheon_p901: everything recorded; CRF eGFR agrees with CKD-EPI(creatinine); no other comorbidity
    assert rows["p002"] == {
        "participant_key": "p002", "site": "incheon", "sex": "F", "makeup": "1", "glasses": "0", "accessories": "1",
        "dm": "0", "htn": "1", "dyslipidemia": "0", "ild": "0", "other_comorbidity_count": "0",
        **_expected_numbers("incheon_p901"),
    }
    # seoul_p903: height above 210 cm, CRF BMI below 13 and DBP above SBP are released as recorded (counted only);
    # CRF eGFR < 15 (agrees with creatinine) counts as renal disease; ild is not ascertained outside incheon
    assert rows["p001"] == {
        "participant_key": "p001", "site": "seoul", "sex": "F", "makeup": "0", "glasses": "1", "accessories": "0",
        "dm": "0", "htn": "1", "dyslipidemia": "0", "ild": "", "other_comorbidity_count": "1",
        **_expected_numbers("seoul_p903"),
    }
    # incheon_p902: FPG 'ND' (missing); CRF eGFR '-' -> computed from creatinine; cerebrovascular disease and cancer
    assert rows["p003"] == {
        "participant_key": "p003", "site": "incheon", "sex": "M", "makeup": "1", "glasses": "1", "accessories": "1",
        "dm": "1", "htn": "0", "dyslipidemia": "1", "ild": "1", "other_comorbidity_count": "2",
        **_expected_numbers("incheon_p902"),
    }
    assert int(rows["p001"]["height"]) > 210 and Decimal(rows["p001"]["dbp"]) > Decimal(rows["p001"]["sbp"])
    blank = {key: [c for c in NUMERIC_COLUMNS if row[c] == ""] for key, row in rows.items()}
    assert blank == {"p001": [], "p002": [], "p003": ["fpg"]}


def test_every_cell_is_an_allowed_category_or_a_number_on_its_grid(fake_source):
    rows = build_participant_table(_crf(fake_source), MAPPING).rows
    for row in rows:
        for column, cell in zip(PARTICIPANT_COLUMNS[1:], row[1:]):
            if column in NUMERIC_COLUMNS:
                assert cell == "" or _is_numeric_cell(column, cell), column
                low, high = PLAUSIBLE_RANGES.get(column, ("0", "1000"))
                assert cell == "" or Decimal(low) <= Decimal(cell) <= Decimal(high), column
            else:
                assert cell in PARTICIPANT_CATEGORIES[column] + ("",), column


CODED = [
    # age and the comorbidity count: whole numbers as recorded (no bands, no top-coding)
    ("age", 57, "57"), ("age", 57.0, "57"), ("age", 19, "19"), ("age", 101, "101"),
    ("other_comorbidity_count", 0.0, "0"), ("other_comorbidity_count", 2.0, "2"), ("other_comorbidity_count", 5, "5"),
    # height to 5 cm and weight to 2 kg, numpy round half to even of round(value, 9) / step
    ("height", 163, "165"), ("height", 157.4, "155"), ("height", 157.5, "160"), ("height", 152.5, "150"),
    ("height", 172.5, "170"), ("height", 98, "100"), ("height", 162.49999999999997, "160"), ("height", 230, "230"),
    ("weight", 71, "72"), ("weight", 73, "72"), ("weight", 70.9, "70"), ("weight", 58, "58"), ("weight", 25, "24"),
    # an already rounded BMI is written with one decimal (the coding is idempotent on released_bmi values)
    ("bmi", 24.2, "24.2"), ("bmi", 25.0, "25.0"), ("bmi", 98.0, "98.0"),
    ("makeup", 0.0, "0"), ("makeup", 1.0, "1"), ("site", "seoul", "seoul"), ("sex", "F", "F"),
]


@pytest.mark.parametrize("column, value, cell", CODED)
def test_codings(column, value, cell):
    assert CODINGS[column].code(value) == cell
    if not isinstance(value, str):
        assert CODINGS[column].code(np.float64(value)) == cell


@pytest.mark.parametrize("column", ["age", "other_comorbidity_count"])
def test_whole_number_columns_refuse_fractions_without_quoting_them(column):
    with pytest.raises(ValueError) as error:
        CODINGS[column].code(57.5)
    assert "57.5" not in str(error.value)


# numpy round half to even of round(value, 9) / step, times step, written with the column's decimals.
ROUNDING = [
    # the 9-decimal snap comes first: float noise just below a half rounds like the half itself
    # (without the snap 17.499999999999996 would give 15, 70.99999999999999 70, 114.99999999999999 110)
    ("ast", 17.499999999999996, "20"), ("gfr", 17.499999999999996, "20"), ("sbp", 17.499999999999996, "20"),
    ("hr_rest", 70.99999999999999, "72"), ("fpg", 114.99999999999999, "120"), ("hba1c", 6.749999999999999, "7.0"),
    # exact halves go to the even multiple
    ("hr_rest", 71, "72"), ("hr_rest", 73, "72"), ("hr_rest", 75, "76"), ("hr_rest", 64, "64"),
    ("fpg", 105, "100"), ("fpg", 115, "120"), ("fpg", 125, "120"), ("fpg", 135, "140"),
    ("sbp", 122.5, "120"), ("sbp", 127.5, "130"), ("dbp", 86, "85"), ("ast", 33, "35"), ("alt", 41, "40"),
    ("hba1c", 6.25, "6.0"), ("hba1c", 6.75, "7.0"), ("gfr", 2.5, "0"), ("gfr", 7.5, "10"),
    ("creatinine", 1.25, "1.2"), ("creatinine", 1.35, "1.4"),
    # binary division: 0.95 / 0.1 = 9.499999999999998 -> 0.9 (an exact decimal rule would give 1.0)
    ("creatinine", 0.95, "0.9"), ("creatinine", 0.15, "0.1"),
    # one decimal for hba1c and creatinine, none for the others
    ("hba1c", 7, "7.0"), ("hba1c", 5.6, "5.5"), ("creatinine", 1, "1.0"), ("creatinine", 2.04, "2.0"),
    ("sbp", 141.0, "140"), ("fpg", 40, "40"),
    # no top or bottom coding: values far out in either tail are only rounded
    ("alt", 1234.0, "1235"), ("ast", 2718.3, "2720"), ("fpg", 612, "610"), ("creatinine", 9.87, "9.9"),
    ("gfr", 0.4, "0"), ("hr_rest", 27, "28"), ("hba1c", 13.3, "13.5"), ("fpg", 12, "10"), ("sbp", 62, "60"),
]


@pytest.mark.parametrize("column, value, cell", ROUNDING)
def test_rounding_rule(column, value, cell):
    assert CODINGS[column].code(value) == cell
    assert CODINGS[column].code(np.float64(value)) == cell


def _series_cells(values, step, decimals):
    """The same rule on a pandas Series (a second, vectorised implementation)."""
    series = pd.Series(values, dtype="float64").round(9)
    rounded = ((series / float(step)).round() * float(step)).round(decimals)
    return ["" if pd.isna(value) else f"{value:.{decimals}f}" for value in rounded]


@pytest.mark.parametrize("column", [c for c in TABLE6_NUMBERS if TABLE6_NUMBERS[c][1] == "round"])
def test_scalar_rounding_equals_the_series_rule(column):
    coding = CODINGS[column]
    step = float(coding.step)
    rng = np.random.default_rng(7)
    halves = (np.arange(400) + 0.5) * step  # exact halves (in binary)
    near = halves * (1 + rng.choice([-1.0, 1.0], halves.size) * 2.0 ** -52)  # float noise around them
    values = np.concatenate([halves, near, rng.uniform(0, 400 * step, 2000), [np.nan]])
    assert [coding.code(value) for value in values] == _series_cells(values, coding.step, coding.decimals)


def _series_bmi_cells(heights, weights):
    """Reference on pandas Series: height and weight rounded first, then weight / (height / 100) ** 2, .round(1)."""
    height = (pd.Series(heights, dtype="float64") / 5).round() * 5
    weight = (pd.Series(weights, dtype="float64") / 2).round() * 2
    bmi = (weight / (height / 100.0) ** 2).round(1)
    return ["" if pd.isna(value) else f"{value:.1f}" for value in bmi]


def test_bmi_is_computed_from_the_rounded_height_and_weight():
    rng = np.random.default_rng(11)
    heights = np.concatenate([rng.uniform(140, 200, 3000), [150, 160, 170, 180, np.nan, 165]])
    weights = np.concatenate([rng.uniform(35, 130, 3000), [45, 64, 70, 90, 70, np.nan]])
    cells = [released_bmi(CODINGS["height"].code(h), CODINGS["weight"].code(w)) for h, w in zip(heights, weights)]
    assert cells == _series_bmi_cells(heights, weights)
    assert cells[-2:] == ["", ""]  # blank when height or weight is blank


@pytest.mark.parametrize("weight, height, bmi", [
    (64, 160, "25.0"),     # 64 / 1.6^2 = 24.999999999999996 in floats; round(1) gives 25.0
    (70, 170, "24.2"),
    (58.88, 160, "22.7"),  # from the rounded weight 58 (the CRF value 58.88 / 1.6^2 would be 23.0)
    (25, 170, "8.3"),      # weight 25 -> 24 kg (half to even)
])
def test_bmi_cells(fake_source, weight, height, bmi):
    rows, _ = _rows_with(fake_source, incheon_p901={"weight": weight, "height": height})
    assert rows["p002"]["bmi"] == bmi


def test_released_bmi_refuses_a_zero_height():
    assert released_bmi("", "70") == "" and released_bmi("170", "") == ""
    with pytest.raises(ValueError):
        released_bmi("0", "70")


@pytest.mark.parametrize("column", PARTICIPANT_COLUMNS[1:])
def test_missing_values_stay_blank(column):
    assert CODINGS[column].code(float("nan")) == ""


def test_codings_reject_inconsistent_definitions_and_unknown_text():
    with pytest.raises(ValueError):
        Coding("x", "bins", ("a", "b"), (1.0,))  # removed in 1.4.0: no column is binned
    with pytest.raises(ValueError):
        Coding("x", "above", ("a", "b"))
    with pytest.raises(ValueError):
        Coding("x", "round")  # no step
    with pytest.raises(ValueError):
        Coding("x", "round", step="0")
    with pytest.raises(ValueError):
        Coding("x", "round", step="0.5", decimals=0)  # 0.5 cannot be written without a decimal
    with pytest.raises(ValueError):
        Coding("x", "round", ("a",), step="5")  # a rounded column has no categories
    with pytest.raises(ValueError):
        Coding("x", "integer", step="5")  # whole numbers have step 1
    with pytest.raises(ValueError):
        Coding("x", "integer", step="1", decimals=1)
    with pytest.raises(ValueError):
        CODINGS["site"].code("busan")


def test_dictionary_is_the_table6_text(fake_source):
    table = build_participant_table(_crf(fake_source), MAPPING)
    assert [row[0] for row in table.dictionary] == list(PARTICIPANT_COLUMNS)
    expected = {column: (description, coding.format(observed=" (observed 0-2)"))
                for column, (description, coding) in TABLE6_TEXT.items()}
    assert {row[0]: (row[2], row[3]) for row in table.dictionary} == expected
    for index, (variable, _group, _description, _coding, available) in enumerate(table.dictionary):
        assert int(available) == sum(1 for row in table.rows if row[index] != ""), variable
    available = {row[0]: row[4] for row in table.dictionary}
    assert (available["ild"], available["fpg"], available["bmi"], available["other_comorbidity_count"]) == \
        ("2", "2", "3", "3")


def test_dictionary_spells_creatinine_and_keeps_the_kept_rule_notes():
    text = " ".join(" ".join(row) for row in DICTIONARY.values())
    assert "Serum creatinine" in text and not re.search(r"creatine\b", text)
    assert "CKD-EPI 2009" in DICTIONARY["gfr"][1] and "when the CRF value is missing" in DICTIONARY["gfr"][1]
    assert DICTIONARY["other_comorbidity_count"][1].endswith("renal disease is also counted when eGFR is below "
                                                             "15 mL/min/1.73 m^2")
    # manuscript Usage Notes: the reference HR comes from BVP_synchronized.csv; HR_synchronized.csv is the device rate
    assert "BVP_synchronized.csv" in DICTIONARY["hr_rest"][1] and "HR_synchronized.csv" in DICTIONARY["hr_rest"][1]
    assert all("acute" not in " ".join(row).lower() for row in DICTIONARY.values())  # ild = interstitial lung disease


def test_dictionary_describes_no_rule_blanking_and_no_former_category():
    for column, (_group, description, coding) in DICTIONARY.items():
        text = f"{description} {coding}".lower()
        assert not any(phrase in text for phrase in RULE_WORDING), column
        assert not any(phrase in text for phrase in CATEGORY_WORDING), column


def test_observed_range_of_the_count_follows_the_rows():
    rows = [[f"p{i:03d}"] + [""] * 23 for i in range(1, 4)]
    count = PARTICIPANT_COLUMNS.index("other_comorbidity_count")
    assert {row[0]: row[3] for row in participants_dictionary(rows)}["other_comorbidity_count"] == "0-5 possible"
    for row, value in zip(rows, ("3", "0", "")):
        row[count] = value
    dictionary = {row[0]: row for row in participants_dictionary(rows)}
    assert dictionary["other_comorbidity_count"][3:] == ["0-5 possible (observed 0-3)", "2"]
    rows[2][count] = ">=1"  # not a whole number (verify reports it): ignored for the observed range, not a crash
    assert {row[0]: row[3] for row in participants_dictionary(rows)}["other_comorbidity_count"] == \
        "0-5 possible (observed 0-3)"


def test_quality_rule_counts(fake_source):
    counts = build_participant_table(_crf(fake_source), MAPPING).quality_counts
    assert list(counts) == QUALITY_COUNT_KEYS
    assert not any(key.endswith("_set_missing") for key in counts)  # 1.4.0: no rule blanks a value
    assert counts == {
        "height_outside_120_210_flagged_unchanged": 1,  # seoul_p903, above 210 cm
        "bmi_outside_13_60_flagged_unchanged": 1,  # seoul_p903 (CRF height and weight, not the rounded ones)
        "sbp_below_70_flagged_unchanged": 0,
        "dbp_not_below_sbp_flagged_unchanged": 1,  # seoul_p903
        "egfr_creatinine_discordant_flagged_unchanged": 0,
        "egfr_computed_from_creatinine": 1,  # incheon_p902
        "ast_isolated_extreme_flagged_unchanged": 0,
        "renal_disease_recoded_from_egfr_below_15": 1,  # seoul_p903
        "fpg_below_40_flagged_unchanged": 0,
        "hr_outside_30_200_flagged_unchanged": 0,
    }


def test_fpg_below_40_is_kept_and_counted(fake_source):
    raw = load_raw(_crf(fake_source), list(MAPPING))
    raw.loc["incheon_p901", "fpg"] = 12
    raw.loc["seoul_p903", "fpg"] = 40  # the threshold itself is not flagged
    data, counts = apply_quality_rules(raw)
    assert counts["fpg_below_40_flagged_unchanged"] == 1 and "fpg_below_40_set_missing" not in counts
    rows = derive_rows(data, MAPPING)
    by_id = _by_id(rows)
    assert (by_id["p002"]["fpg"], by_id["p001"]["fpg"]) == ("10", "40")  # rounded, not blanked
    fpg = {row[0]: row for row in participants_dictionary(rows)}["fpg"]
    assert fpg[4] == "2"  # p003 has no FPG; p002 keeps its value
    assert raw.loc["incheon_p901", "fpg"] == 12  # the raw frame is not modified


# Since 1.4.0 the values the 1.3.0 rules removed are released. Per former rule: CRF edits for incheon_p901 (release
# ID p002), the cells it now gets, and the count that records the hit (no other count changes).
FORMER_BLANKING_RULES = [
    ({"height": 230.0, "weight": 70.0}, {"height": "230", "bmi": "13.2"}, "height_outside_120_210_flagged_unchanged"),
    ({"height": 110.0, "weight": 70.0}, {"height": "110", "bmi": "57.9"}, "height_outside_120_210_flagged_unchanged"),
    ({"weight": 25.0, "height": 170.0}, {"weight": "24", "bmi": "8.3"}, "bmi_outside_13_60_flagged_unchanged"),
    ({"sbp": 62.0, "dbp": 40.0}, {"sbp": "60", "dbp": "40"}, "sbp_below_70_flagged_unchanged"),
    ({"sbp": 125.0, "dbp": 130.0}, {"sbp": "125", "dbp": "130"}, "dbp_not_below_sbp_flagged_unchanged"),
    ({"creatinine": 0.9, "egfr": 30.0}, {"creatinine": "0.9", "gfr": "30"},
     "egfr_creatinine_discordant_flagged_unchanged"),
    ({"ast": 400.0, "alt": 20.0}, {"ast": "400", "alt": "20"}, "ast_isolated_extreme_flagged_unchanged"),
    ({"fpg": 12.0}, {"fpg": "10"}, "fpg_below_40_flagged_unchanged"),
]


@pytest.mark.parametrize("edits, cells, key", FORMER_BLANKING_RULES)
def test_former_blanking_rules_only_count(fake_source, edits, cells, key):
    raw = load_raw(_crf(fake_source), list(MAPPING))
    _, baseline = apply_quality_rules(raw)
    for column, value in edits.items():
        raw.loc["incheon_p901", column] = value
    data, counts = apply_quality_rules(raw)
    row = _by_id(derive_rows(data, MAPPING))["p002"]
    assert {column: row[column] for column in cells} == cells
    assert counts == {**baseline, key: baseline[key] + 1}
    assert all(raw.loc["incheon_p901", column] == value for column, value in edits.items())


def test_no_rule_blanks_a_recorded_value(fake_source):
    """A released cell is blank only when the CRF value is missing (gfr: also when it cannot be computed)."""
    raw = load_raw(_crf(fake_source), list(MAPPING))
    raw.loc["incheon_p901", ["height", "sbp", "dbp", "egfr", "ast", "alt", "fpg"]] = [100, 60, 65, 10, 900, 30, 5]
    data, _ = apply_quality_rules(raw)
    rows = _by_id(derive_rows(data, MAPPING))
    for key, release_id in MAPPING.items():
        for column in ("age", "height", "weight", "sbp", "dbp", "hr_rest", "hba1c", "fpg", "creatinine", "ast", "alt"):
            assert (rows[release_id][column] == "") == pd.isna(raw.loc[key, column]), (release_id, column)
        bmi_known = pd.notna(raw.loc[key, "height"]) and pd.notna(raw.loc[key, "weight"])
        assert (rows[release_id]["bmi"] != "") == bmi_known, release_id
        gfr_known = pd.notna(raw.loc[key, "egfr"]) or pd.notna(raw.loc[key, "creatinine"])
        assert (rows[release_id]["gfr"] != "") == gfr_known, release_id
    assert (rows["p002"]["height"], rows["p002"]["fpg"], rows["p002"]["gfr"], rows["p002"]["ast"]) == \
        ("100", "0", "10", "900")


def test_egfr_is_still_computed_from_creatinine(fake_source):
    """Kept: CRF eGFR missing and creatinine present -> CKD-EPI 2009 (information added, not removed)."""
    crf = SYNTHETIC_CRF[SCREENING["incheon_p902"]]
    expected = reference_ckd_epi_2009(crf["Creatinine"], crf["Age"], crf["Sex"] == FEMALE)
    raw = load_raw(_crf(fake_source), list(MAPPING))
    data, counts = apply_quality_rules(raw)
    assert counts["egfr_computed_from_creatinine"] == 1
    assert math.isclose(data.loc["incheon_p902", "egfr"], expected, rel_tol=1e-9)
    assert _by_id(derive_rows(data, MAPPING))["p003"]["gfr"] == _reference_cell(expected, "5", 0)
    raw.loc["incheon_p902", "creatinine"] = np.nan  # nothing to compute from: gfr stays blank
    data, counts = apply_quality_rules(raw)
    assert counts["egfr_computed_from_creatinine"] == 0
    assert _by_id(derive_rows(data, MAPPING))["p003"]["gfr"] == ""


def test_egfr_below_15_still_counts_as_renal_disease(fake_source):
    """Kept: eGFR < 15 -> renal disease in other_comorbidity_count, also for an eGFR that disagrees with creatinine."""
    raw = load_raw(_crf(fake_source), list(MAPPING))
    data, counts = apply_quality_rules(raw)
    assert counts["renal_disease_recoded_from_egfr_below_15"] == 1
    assert _by_id(derive_rows(data, MAPPING))["p001"]["other_comorbidity_count"] == "1"  # only via eGFR < 15
    raw.loc["incheon_p901", "egfr"] = 10.0  # far from CKD-EPI(creatinine): discordant, kept, and below 15
    data, counts = apply_quality_rules(raw)
    assert counts["renal_disease_recoded_from_egfr_below_15"] == 2
    assert counts["egfr_creatinine_discordant_flagged_unchanged"] == 1
    rows = _by_id(derive_rows(data, MAPPING))
    assert (rows["p002"]["gfr"], rows["p002"]["other_comorbidity_count"]) == ("10", "1")
    raw.loc["seoul_p903", "egfr"] = 15.0  # the threshold itself is not kidney failure
    data, counts = apply_quality_rules(raw)
    assert counts["renal_disease_recoded_from_egfr_below_15"] == 1
    assert _by_id(derive_rows(data, MAPPING))["p001"]["other_comorbidity_count"] == "0"


def test_renal_disease_already_recorded_is_not_counted_twice(fake_source):
    rows, counts = _rows_with(fake_source, seoul_p903={"other_renal": 1.0})
    assert counts["renal_disease_recoded_from_egfr_below_15"] == 0
    assert rows["p001"]["other_comorbidity_count"] == "1"


def test_unknown_renal_item_with_egfr_below_15_counts_as_renal_disease(fake_source):
    rows, counts = _rows_with(fake_source, seoul_p903={"other_renal": np.nan})
    assert counts["renal_disease_recoded_from_egfr_below_15"] == 1
    assert rows["p001"]["other_comorbidity_count"] == "1"


def test_implausible_hr_is_counted_but_not_changed(fake_source):
    raw = load_raw(_crf(fake_source), list(MAPPING))
    raw.loc["incheon_p901", "hr_rest"] = 220
    data, counts = apply_quality_rules(raw)
    assert counts["hr_outside_30_200_flagged_unchanged"] == 1
    assert _by_id(derive_rows(data, MAPPING))["p002"]["hr_rest"] == "220"


def test_other_comorbidity_count_is_the_number_of_positive_items(fake_source):
    """The sum of the five 0/1 items, 0-5, no top-coding."""
    rows, _ = _rows_with(fake_source, incheon_p901={"other_cardiovascular": 1.0, "other_cancer": 1.0,
                                                    "other_lung": 1.0})
    assert rows["p002"]["other_comorbidity_count"] == "3"
    names = ["other_cardiovascular", "other_cerebrovascular", "other_cancer", "other_renal", "other_lung"]
    frame = pd.DataFrame([[1.0] * 5, [0.0] * 5, [1.0, 0.0, 1.0, 0.0, 0.0]], columns=names)
    assert other_comorbidity_count(frame).tolist() == [5.0, 0.0, 2.0]


def test_other_comorbidity_count_is_blank_when_an_item_is_unknown(fake_source):
    """A sum with min_count=5: blank unless all five items are known, even when some are present."""
    rows, _ = _rows_with(fake_source, incheon_p902={"other_lung": np.nan}, incheon_p901={"other_lung": np.nan})
    assert rows["p003"]["other_comorbidity_count"] == ""  # two present, lung unknown
    assert rows["p002"]["other_comorbidity_count"] == ""  # four absent, one unknown


def test_quality_rules_do_not_modify_the_raw_frame(fake_source):
    raw = load_raw(_crf(fake_source), list(MAPPING))
    raw.loc["incheon_p901", "fpg"] = 35
    before = raw.copy()
    apply_quality_rules(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_crf_workbooks_are_only_read(fake_source):
    before = tree_fingerprint(fake_source["root"]), fake_source["ild_xlsx"].read_bytes()
    build_participant_table(_crf(fake_source), MAPPING)
    assert (tree_fingerprint(fake_source["root"]), fake_source["ild_xlsx"].read_bytes()) == before


def test_participant_without_crf_row_is_an_error(fake_source):
    with pytest.raises(ValueError):
        build_participant_table(_crf(fake_source), {**MAPPING, "seoul_p999": "p004"})


def test_fractional_crf_age_is_an_error(fake_source):
    raw = load_raw(_crf(fake_source), list(MAPPING))
    raw.loc["incheon_p901", "age"] = 57.5
    data, _ = apply_quality_rules(raw)
    with pytest.raises(ValueError):
        derive_rows(data, MAPPING)


def test_ild_outside_incheon_is_an_error(fake_source, tmp_path):
    bad = tmp_path / "ild_bad.xlsx"
    _write_sheet(bad, ILD_SHEET, ["Subject Screening NO", "ILD"], [["04-S901", "O"], ["01-S903", "O"]])
    with pytest.raises(ValueError):
        build_participant_table(CrfSources.for_source(fake_source["root"], ild_xlsx=bad), MAPPING)


def test_incheon_participant_without_ild_value_is_an_error(fake_source, tmp_path):
    bad = tmp_path / "ild_missing.xlsx"
    _write_sheet(bad, ILD_SHEET, ["Subject Screening NO", "ILD"], [["04-S901", "O"], ["04-S902", "-"]])
    with pytest.raises(ValueError):
        build_participant_table(CrfSources.for_source(fake_source["root"], ild_xlsx=bad), MAPPING)


def test_sheets_are_read_by_name_or_as_the_first_sheet(fake_source):
    first = dataclasses.replace(_crf(fake_source), crf_sheet=None, ild_sheet=None)  # default without KRRPPG_*_SHEET
    named = dataclasses.replace(first, crf_sheet=CRF_SHEET, ild_sheet=ILD_SHEET)
    assert build_participant_table(named, MAPPING) == build_participant_table(first, MAPPING)
    with pytest.raises(ValueError):
        build_participant_table(dataclasses.replace(first, crf_sheet="no such sheet"), MAPPING)


def test_ckd_epi_2009_reference_values():
    # synthetic inputs covering both branches: creatinine above kappa (male) and below kappa (female)
    egfr = ckd_epi_2009(pd.Series([1.35, 0.55]), pd.Series([61, 47]), pd.Series(["M", "F"]))
    assert math.isclose(egfr[0], 141 * (1.35 / 0.9) ** -1.209 * 0.993 ** 61, rel_tol=1e-9)
    assert math.isclose(egfr[1], 141 * (0.55 / 0.7) ** -0.329 * 0.993 ** 47 * 1.018, rel_tol=1e-9)
    assert math.isclose(ckd_epi_2009(pd.Series([0.9]), pd.Series([40]), pd.Series(["M"]))[0],
                        141 * 0.993 ** 40, rel_tol=1e-9)  # ratio exactly 1: both power terms are 1

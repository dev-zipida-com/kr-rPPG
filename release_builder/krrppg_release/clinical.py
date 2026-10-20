"""Participant metadata for the release, derived from raw CRF values.

KR-rPPG/metadata/participants.csv keeps the 24 variables of manuscript Table 6, with the same names and order,
coded as manuscript Table 6 describes them (since 1.4.0; the categories of 1.2.x-1.3.0 are no longer used):

    participant_key  random release ID pNNN
    site             site letter A / B / C     sex              as recorded
    age              years (whole number, as in the CRF)
    height           rounded to 5 cm           weight           rounded to 2 kg
    bmi              weight / (height / 100)^2 of the rounded height and weight, one decimal (released_bmi)
    makeup ... ild   0/1 flags (ild: site C only, blank = not ascertained)
    sbp, dbp         rounded to 5 mmHg         hr_rest          rounded to 2 bpm
    hba1c            rounded to 0.5 %          fpg              rounded to 10 mg/dL
    creatinine       rounded to 0.1 mg/dL      gfr              rounded to 5 mL/min/1.73 m^2
    ast, alt         rounded to 5 U/L
    other_comorbidity_count  number of positive items among five (0-5, no top-coding)

site, sex and the 0/1 flags are the only categorical columns; the other columns are numbers (age in whole years,
the rest rounded to a fixed step), so they can hold rare values, and combinations of columns are not k-anonymous.
The release is distributed openly (CC BY-NC 4.0), so the residual risk is addressed only by the responsible-use
conditions published with it. There is no top or bottom coding. Everything is
computed from the raw CRF values (not from the rounded <source_root>/metadata/participants.csv). No data-quality
check removes a value: values the 1.3.0 rules blanked are released as recorded in the CRF and the check hits are
only counted; the two in-memory corrections that add information are kept (eGFR computed with CKD-EPI 2009 from
creatinine when the CRF eGFR is missing, eGFR < 15 counted as renal disease). The CRF workbooks are opened
read-only and never written.

Since 1.5.0 the collection sites are named only by the letters A, B and C (SITE_PREFIX maps the CRF screening-
number prefix to the letter), in participants.csv, in the dictionary and in the private `site_pNNN` keys.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# CRF workbooks of the data custodian (never released). Locations come from the CLI (--crf, --crf-ild) or from
# the environment; sheet names from KRRPPG_CRF_SHEET / KRRPPG_ILD_SHEET, otherwise the first sheet is read.
DEFAULT_ILD_XLSX = Path(os.environ["KRRPPG_ILD_XLSX"]) if os.environ.get("KRRPPG_ILD_XLSX") else None
CRF_FILENAME = os.environ.get("KRRPPG_CRF_FILENAME", "crf.xlsx")
SCREENING_COL = "Subject Screening NO"
# CRF screening-number prefix -> collection site letter (released as is in the site column since 1.5.0).
SITE_PREFIX = {"01": "A", "03": "B", "04": "C"}
SITES = ("A", "B", "C")
ILD_SITE = "C"  # the CRF ILD item was ascertained at this site only
MISSING_TOKENS = ["-", "ND", "NA", "N/A", "na", "nan", "None", ""]
PRESENT, ABSENT = 1, 2  # CRF coding for binary items
RAW_NUMERIC = {
    "age": "Age", "height": "Height", "weight": "Weight", "sbp": "SBP", "dbp": "DBP", "hr_rest": "HR",
    "hba1c": "HbA1c", "fpg": "FPG", "creatinine": "Creatinine", "egfr": "GFR(CKD-EPI)", "ast": "AST", "alt": "ALT",
}
RAW_BINARY = {
    "makeup": "Makeup", "glasses": "Glasses", "accessories": "Accessories",
    "dm": "DM", "htn": "HTN", "dyslipidemia": "Dyslipidemia",
}
OTHER_COMORBIDITIES = {
    "cardiovascular": "Cardiovascular", "cerebrovascular": "Cerebrovascular", "cancer": "Cancer",
    "renal": "Renal disease", "lung": "Lung disease",
}

# Plausibility checks on the CRF values, counted in the build report ('..._flagged_unchanged'); the values are
# released as recorded.
# Up to 1.3.0 all but the HR check blanked the value; since 1.4.0 no check removes a value.
HEIGHT_RANGE_CM = (120, 210)
BMI_RANGE = (13, 60)
SBP_MIN_PLAUSIBLE = 70
EGFR_DISCORDANCE = 15  # |recorded eGFR - CKD-EPI 2009(creatinine)| in mL/min/1.73 m^2
AST_ISOLATED = (200, 10)  # AST >= 200 U/L with AST/ALT >= 10
FPG_MIN_PLAUSIBLE = 40  # mg/dL
HR_FLAG_RANGE = (30, 200)  # bpm
# Correction that adds information (kept): eGFR below this counts as renal disease in other_comorbidity_count.
EGFR_KIDNEY_FAILURE = 15
SNAP_DECIMALS = 9  # removes float noise (17.499999999999996 -> 17.5) before rounding to a step

# Inclusive range of every released value of a numeric column except bmi; verify reports values outside as errors.
# Wide limits that catch unit or column mix-ups (age as a birth year, height in m or mm, weight in g, creatinine in
# umol/L, FPG in mmol/L, HbA1c in mmol/mol), written on each column's grid. They are not data-quality rules and
# lie well outside the former ones: since 1.4.0 the values below SBP_MIN_PLAUSIBLE and FPG_MIN_PLAUSIBLE and the
# heights outside HEIGHT_RANGE_CM are released, so the sbp and fpg lower bounds are no longer the former thresholds
# (70 and 40 in 1.3.0). The comorbidity count has five items. bmi has no range of its own: verify checks it against
# the row's height and weight (released_bmi).
PLAUSIBLE_RANGES: Dict[str, Tuple[str, str]] = {
    "age": ("0", "120"), "height": ("50", "250"), "weight": ("10", "300"),
    "sbp": ("50", "300"), "dbp": ("20", "200"), "hr_rest": ("20", "250"), "hba1c": ("3.0", "20.0"),
    "fpg": ("10", "1000"), "creatinine": ("0.1", "20.0"), "gfr": ("0", "200"), "ast": ("0", "5000"),
    "alt": ("0", "5000"), "other_comorbidity_count": ("0", str(len(OTHER_COMORBIDITIES))),
}


@dataclass(frozen=True)
class CrfSources:
    crf_xlsx: Path
    crf_sheet: Optional[str] = os.environ.get("KRRPPG_CRF_SHEET") or None  # None: the first sheet
    ild_xlsx: Optional[Path] = DEFAULT_ILD_XLSX
    ild_sheet: Optional[str] = os.environ.get("KRRPPG_ILD_SHEET") or None
    header_row: int = 5

    @classmethod
    def for_source(cls, source_root, crf_xlsx=None, ild_xlsx=None) -> CrfSources:
        return cls(
            crf_xlsx=Path(crf_xlsx) if crf_xlsx else Path(source_root) / CRF_FILENAME,
            ild_xlsx=Path(ild_xlsx) if ild_xlsx else DEFAULT_ILD_XLSX,
        )

    def files(self) -> Tuple[Path, ...]:
        """The configured workbooks (the ILD workbook may be unset when no metadata is rendered)."""
        return tuple(path for path in (self.crf_xlsx, self.ild_xlsx) if path is not None)


# --- column codings ----------------------------------------------------------------------

CODING_KINDS = ("text", "flag", "integer", "round")
NUMERIC_KINDS = ("integer", "round")  # written as numbers: no categories, checked on their grid by verify


@dataclass(frozen=True)
class Coding:
    """How one released column is coded from one value ('' = missing).

    kind "text":    categorical text kept as is (one of `categories`)
    kind "flag":    0/1 indicator
    kind "integer": a whole number written without decimals (step "1"); a fraction is an error, never rounded
    kind "round":   the number rounded to a multiple of `step` (decimal text such as "0.5") and written with
                    `decimals` decimals (round_to_step); no categories
    """

    source: str  # column of the frame derive_rows codes (apply_quality_rules output, plus other_comorbidity_count)
    kind: str
    categories: Tuple[str, ...] = ()
    step: Optional[str] = None
    decimals: int = 0

    def __post_init__(self):
        if self.kind not in CODING_KINDS:
            raise ValueError(f"unknown coding kind {self.kind!r}")
        if self.kind in NUMERIC_KINDS and not _valid_rounding(self.step, self.decimals, self.categories):
            raise ValueError(f"{self.source}: a numeric coding needs a positive decimal step that can be written "
                             f"with {self.decimals} decimals, and no categories")
        if self.kind == "integer" and (self.step != "1" or self.decimals != 0):
            raise ValueError(f"{self.source}: whole numbers have step 1 and no decimals")

    def code(self, value) -> str:
        if pd.isna(value):
            return ""
        if self.kind == "text":
            if value not in self.categories:
                raise ValueError(f"{self.source}: a value outside {self.categories}")
            return str(value)
        if self.kind == "flag":
            return _flag(value)
        if self.kind == "integer":
            if not float(value).is_integer():
                raise ValueError(f"{self.source}: a value that is not a whole number")  # the value is not quoted
            return str(int(value))
        return round_to_step(value, float(self.step), self.decimals)


def _valid_rounding(step: Optional[str], decimals: int, categories: Tuple[str, ...]) -> bool:
    try:
        increment = Decimal(step)
    except (TypeError, ValueError, ArithmeticError):  # None or not a number
        return False
    return (increment.is_finite() and increment > 0 and decimals >= 0 and not categories
            and increment % Decimal(1).scaleb(-decimals) == 0)


def _flag_coding(source: str) -> Coding:
    return Coding(source, "flag", ("0", "1"))


def _integer(source: str) -> Coding:
    return Coding(source, "integer", step="1")


def _rounded(source: str, step: str, decimals: int = 0) -> Coding:
    return Coding(source, "round", step=step, decimals=decimals)


PARTICIPANT_COLUMNS = (
    "participant_key", "site", "age", "sex", "height", "weight", "bmi",
    "makeup", "glasses", "accessories", "sbp", "dbp", "hr_rest",
    "hba1c", "fpg", "creatinine", "gfr", "ast", "alt",
    "dm", "htn", "dyslipidemia", "ild", "other_comorbidity_count",
)
# Manuscript Table 6 coding of every column (whole numbers, rounding steps and decimals).
CODINGS: Dict[str, Coding] = {
    "site": Coding("site", "text", SITES),
    "age": _integer("age"),
    "sex": Coding("sex", "text", ("M", "F")),
    "height": _rounded("height", "5"),
    "weight": _rounded("weight", "2"),
    # derive_rows writes released_bmi(height cell, weight cell); this coding is its grid (idempotent on it)
    "bmi": _rounded("bmi", "0.1", 1),
    "makeup": _flag_coding("makeup"),
    "glasses": _flag_coding("glasses"),
    "accessories": _flag_coding("accessories"),
    # the nine clinical measurements: rounding step and decimals
    "sbp": _rounded("sbp", "5"),
    "dbp": _rounded("dbp", "5"),
    "hr_rest": _rounded("hr_rest", "2"),
    "hba1c": _rounded("hba1c", "0.5", 1),
    "fpg": _rounded("fpg", "10"),
    "creatinine": _rounded("creatinine", "0.1", 1),
    "gfr": _rounded("egfr", "5"),
    "ast": _rounded("ast", "5"),
    "alt": _rounded("alt", "5"),
    "dm": _flag_coding("dm"),
    "htn": _flag_coding("htn"),
    "dyslipidemia": _flag_coding("dyslipidemia"),
    "ild": _flag_coding("interstitial_lung_disease"),
    "other_comorbidity_count": _integer("other_comorbidity_count"),
}
NUMERIC_COLUMNS = tuple(name for name in PARTICIPANT_COLUMNS[1:] if CODINGS[name].kind in NUMERIC_KINDS)
CATEGORICAL_COLUMNS = tuple(name for name in PARTICIPANT_COLUMNS[1:] if name not in NUMERIC_COLUMNS)
# Every value a categorical participants.csv column (site, sex, the 0/1 flags) may hold besides '' (missing);
# participant_key is pNNN and the numeric columns hold numbers (Coding.step, Coding.decimals, PLAUSIBLE_RANGES).
PARTICIPANT_CATEGORIES: Dict[str, Tuple[str, ...]] = {name: CODINGS[name].categories for name in CATEGORICAL_COLUMNS}


# --- reading ---------------------------------------------------------------------------

def screening_to_key(value) -> Optional[str]:
    """`04-S906` / `03-s904` -> `C_p906` / `B_p904`."""
    match = re.match(r"^(\d{2})[-_]s(\d+)$", str(value).strip(), re.IGNORECASE)
    if not match or match.group(1) not in SITE_PREFIX:
        return None
    return f"{SITE_PREFIX[match.group(1)]}_p{int(match.group(2)):03d}"


def _numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    # float64 even for all-integer columns, so every numeric column has one dtype (NaN = missing)
    return pd.to_numeric(text.where(~text.isin(MISSING_TOKENS)), errors="coerce").astype("float64")


def _binary(series: pd.Series) -> pd.Series:
    return _numeric(series).map({PRESENT: 1.0, ABSENT: 0.0})


def _keyed_sheet(path: Path, sheet: Optional[str], header_row: int) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=0 if sheet is None else sheet, header=header_row)  # 0: the first sheet
    keys = frame[SCREENING_COL].map(screening_to_key)
    frame = frame.assign(participant_key=keys).dropna(subset=["participant_key"])
    duplicated = frame["participant_key"].duplicated()
    if duplicated.any():
        where = f"{path} [{sheet or 'first sheet'}]"
        raise ValueError(f"{where}: duplicate participants {sorted(frame.loc[duplicated, 'participant_key'])[:5]}")
    return frame.set_index("participant_key")


def load_raw(sources: CrfSources, keys: Sequence[str]) -> pd.DataFrame:
    """Raw CRF values for exactly `keys` (read-only)."""
    crf = _keyed_sheet(sources.crf_xlsx, sources.crf_sheet, sources.header_row)
    missing = sorted(set(keys) - set(crf.index))
    if missing:
        raise ValueError(f"{len(missing)} released participants have no CRF row: {missing[:5]}")
    crf = crf.loc[sorted(keys)]
    raw = pd.DataFrame(index=crf.index)
    raw["site"] = [key.split("_")[0] for key in crf.index]
    raw["sex"] = _numeric(crf["Sex"]).map({1: "M", 2: "F"})
    for name, column in RAW_NUMERIC.items():
        raw[name] = _numeric(crf[column])
    for name, column in RAW_BINARY.items():
        raw[name] = _binary(crf[column])
    for name, column in OTHER_COMORBIDITIES.items():
        raw[f"other_{name}"] = _binary(crf[column])

    if sources.ild_xlsx is None:
        raise ValueError("no ILD workbook: pass --crf-ild or set KRRPPG_ILD_XLSX")
    ild = _keyed_sheet(sources.ild_xlsx, sources.ild_sheet, sources.header_row)["ILD"]
    coded = ild.astype(str).str.strip().str.upper().map({"O": 1.0, "X": 0.0})
    raw["interstitial_lung_disease"] = coded.reindex(raw.index)
    uncovered = raw.loc[raw["site"] == ILD_SITE, "interstitial_lung_disease"].isna()
    if uncovered.any():
        raise ValueError(f"{int(uncovered.sum())} site {ILD_SITE} participants have no O/X value for CRF ILD")
    outside = raw.loc[raw["site"] != ILD_SITE, "interstitial_lung_disease"].notna()
    if outside.any():
        raise ValueError(f"interstitial lung disease (CRF ILD) is recorded outside site {ILD_SITE}")
    return raw


# --- data quality ----------------------------------------------------------------------

def ckd_epi_2009(creatinine: pd.Series, age: pd.Series, sex: pd.Series) -> pd.Series:
    """CKD-EPI 2009 creatinine equation (no race coefficient)."""
    female = (sex == "F").to_numpy()
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.329, -0.411)
    ratio = creatinine.to_numpy(dtype=float) / kappa
    egfr = 141 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.209 * 0.993 ** age.to_numpy(dtype=float)
    egfr = np.where(female, egfr * 1.018, egfr)
    valid = creatinine.notna() & age.notna() & sex.notna()
    return pd.Series(egfr, index=creatinine.index).where(valid)


def _outside(values: pd.Series, bounds: Tuple[float, float]) -> pd.Series:
    return (values < bounds[0]) | (values > bounds[1])


def apply_quality_rules(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Return a copy with the derived values and per-check counts (aggregate only; `raw` is not modified).

    No check removes a value (1.4.0): the former blanking rules are counted as '..._flagged_unchanged'
    on the CRF values (the BMI check on the unrounded CRF height and weight, as before). Added: eGFR computed
    with CKD-EPI 2009 when the CRF eGFR is missing, and renal disease when eGFR is below 15 mL/min/1.73 m^2.
    The released bmi is computed later from the rounded height and weight (released_bmi).
    """
    data = raw.copy()
    counts = {}

    bmi = data["weight"] / (data["height"] / 100) ** 2
    counts["height_outside_120_210_flagged_unchanged"] = int(_outside(data["height"], HEIGHT_RANGE_CM).sum())
    counts["bmi_outside_13_60_flagged_unchanged"] = int(_outside(bmi, BMI_RANGE).sum())
    counts["sbp_below_70_flagged_unchanged"] = int((data["sbp"] < SBP_MIN_PLAUSIBLE).sum())
    counts["dbp_not_below_sbp_flagged_unchanged"] = int((data["dbp"] >= data["sbp"]).sum())

    computed = ckd_epi_2009(data["creatinine"], data["age"], data["sex"])
    discordant = (data["egfr"] - computed).abs() > EGFR_DISCORDANCE  # False where either value is missing
    counts["egfr_creatinine_discordant_flagged_unchanged"] = int(discordant.sum())
    fill = data["egfr"].isna() & computed.notna()
    counts["egfr_computed_from_creatinine"] = int(fill.sum())
    data.loc[fill, "egfr"] = computed[fill]

    isolated_ast = (data["ast"] >= AST_ISOLATED[0]) & ((data["ast"] / data["alt"]) >= AST_ISOLATED[1])
    counts["ast_isolated_extreme_flagged_unchanged"] = int(isolated_ast.sum())

    kidney_failure = (data["egfr"] < EGFR_KIDNEY_FAILURE) & (data["other_renal"] != 1)
    counts["renal_disease_recoded_from_egfr_below_15"] = int(kidney_failure.sum())
    data.loc[kidney_failure, "other_renal"] = 1.0

    counts["fpg_below_40_flagged_unchanged"] = int((data["fpg"] < FPG_MIN_PLAUSIBLE).sum())
    counts["hr_outside_30_200_flagged_unchanged"] = int(_outside(data["hr_rest"], HR_FLAG_RANGE).sum())
    return data, counts


# --- derivation --------------------------------------------------------------------------

def round_snap(value: float) -> float:
    """The 9-decimal float-noise snap (numpy rounding) applied before rounding to a step."""
    return float(np.round(np.float64(value), SNAP_DECIMALS))


def round_to_step(value: float, step: float, decimals: int) -> str:
    """One value rounded to a multiple of `step` and written with `decimals` decimals.

    numpy round half to even of round(value, 9) / step, times step, rounded to `decimals`.
    The snap comes first (17.499999999999996 -> 17.5 -> 20 for step 5); the division is binary floating
    point, so a decimal half that is stored just below the half goes down (0.95 / 0.1 -> 0.9).
    """
    multiple = np.round(round_snap(value) / step) * step
    return f"{np.round(multiple, decimals):.{decimals}f}"


def released_bmi(height: str, weight: str) -> str:
    """The bmi cell from the released height and weight cells ('' when either is blank).

    Manuscript Table 6 ("Body mass index from rounded height/weight"):
    numpy round(weight / (height / 100) ** 2, 1) of the rounded numbers, written with one decimal. The bmi
    coding (step 0.1, one decimal) leaves such a value unchanged. verify uses this function to check the
    released rows.
    """
    if height == "" or weight == "":
        return ""
    height_m = float(height) / 100.0
    if not height_m > 0:
        raise ValueError("bmi: the height cell is not a positive number")  # the value is not quoted
    return CODINGS["bmi"].code(np.round(float(weight) / height_m ** 2, 1))


def _flag(value) -> str:
    return "" if pd.isna(value) else str(int(value))


def other_comorbidity_count(data: pd.DataFrame) -> pd.Series:
    """Number of the five other comorbidities present (0-5); NaN unless all five items are known (a sum with
    min_count=5). Renal disease includes the eGFR < 15 correction of apply_quality_rules."""
    other = data[[f"other_{name}" for name in OTHER_COMORBIDITIES]]
    return other.sum(axis=1, min_count=len(OTHER_COMORBIDITIES))


def derive_rows(data: pd.DataFrame, mapping: Dict[str, str]) -> List[List[str]]:
    """participants.csv rows (PARTICIPANT_COLUMNS), sorted by release ID."""
    data = data.assign(other_comorbidity_count=other_comorbidity_count(data))
    rows = []
    for key, row in data.iterrows():
        cells = {column: coding.code(row[coding.source]) for column, coding in CODINGS.items() if column != "bmi"}
        cells["bmi"] = released_bmi(cells["height"], cells["weight"])
        rows.append([mapping[key]] + [cells[column] for column in PARTICIPANT_COLUMNS[1:]])
    return sorted(rows)


# --- dictionary ----------------------------------------------------------------------------

# Manuscript Table 6: description and unit / coding of every column (spelled 'creatinine'; units written with
# m^2). Added to the Table 6 text: the two kept corrections (gfr, other_comorbidity_count) and the 1.2.1 hr_rest
# clarification; since 1.5.0 site and ild name the site letters. '{observed}' is replaced by the observed range of
# the count (participants_dictionary).
DICTIONARY = {
    "participant_key": ("Identifier", "Pseudonymized participant identifier (pNNN; e.g., p001)", "-"),
    "site": ("Demographics", "Collection site (A / B / C)", "-"),
    "age": ("Demographics", "Age at enrollment", "years"),
    "sex": ("Demographics", "Biological sex", "M/F"),
    "height": ("Demographics", "Body height", "cm, rounded to 5"),
    "weight": ("Demographics", "Body weight", "kg, rounded to 2"),
    "bmi": ("Demographics", "Body mass index from rounded height/weight", "kg/m^2, 1 decimal"),
    "makeup": ("Acquisition appearance indicators", "Facial makeup present at recording", "0/1"),
    "glasses": ("Acquisition appearance indicators", "Eyeglasses worn at recording", "0/1"),
    "accessories": ("Acquisition appearance indicators", "Facial accessories worn at recording", "0/1"),
    "sbp": ("Clinical measurements", "Systolic blood pressure", "mmHg, rounded to 5"),
    "dbp": ("Clinical measurements", "Diastolic blood pressure", "mmHg, rounded to 5"),
    "hr_rest": (
        "Clinical measurements",
        ("Resting heart rate (CRF vital sign; not the benchmark reference HR, which is derived from the contact pulse "
         "waveform in each session's BVP_synchronized.csv, nor the CMS50E device pulse rate in HR_synchronized.csv)"),
        "bpm, rounded to 2",
    ),
    "hba1c": ("Clinical measurements", "Glycated hemoglobin", "%, rounded to 0.5"),
    "fpg": ("Clinical measurements", "Fasting plasma glucose", "mg/dL, rounded to 10"),
    "creatinine": ("Clinical measurements", "Serum creatinine", "mg/dL, rounded to 0.1"),
    "gfr": (
        "Clinical measurements",
        ("eGFR (CKD-EPI); computed from serum creatinine with the CKD-EPI 2009 equation (no race coefficient) when "
         "the CRF value is missing"),
        "mL/min/1.73 m^2, rounded to 5",
    ),
    "ast": ("Clinical measurements", "Aspartate aminotransferase", "U/L, rounded to 5"),
    "alt": ("Clinical measurements", "Alanine aminotransferase", "U/L, rounded to 5"),
    "dm": ("Comorbidity indicators", "Diabetes mellitus", "0/1"),
    "htn": ("Comorbidity indicators", "Hypertension", "0/1"),
    "dyslipidemia": ("Comorbidity indicators", "Dyslipidemia", "0/1"),
    "ild": ("Comorbidity indicators", "Interstitial lung disease (site C only; blank = not ascertained)", "0/1"),
    "other_comorbidity_count": (
        "Comorbidity indicators",
        ("Number of positive indicators among cardiovascular disease, cerebrovascular disease, cancer, renal disease, "
         "and lung disease; renal disease is also counted when eGFR is below 15 mL/min/1.73 m^2"),
        f"0-{len(OTHER_COMORBIDITIES)} possible{{observed}}",
    ),
}


def dictionary_header(n: int) -> List[str]:
    return ["variable", "variable_group", "description", "unit_or_coding", f"availability_n_of_{n}"]


def _observed_range(cells: Sequence[str]) -> str:
    """' (observed a-b)' over the whole-number cells ('' when there are none; other text is left to verify)."""
    values = sorted(int(cell) for cell in cells if cell.isascii() and cell.isdigit())
    return f" (observed {values[0]}-{values[-1]})" if values else ""


def participants_dictionary(rows: List[List[str]]) -> List[List[str]]:
    """participants_dictionary.csv rows; a pure function of the participants.csv rows."""
    out = []
    for index, column in enumerate(PARTICIPANT_COLUMNS):
        group, description, coding = DICTIONARY[column]
        cells = [row[index] for row in rows]
        coding = coding.format(observed=_observed_range(cells)) if "{observed}" in coding else coding
        out.append([column, group, description, coding, str(sum(1 for cell in cells if cell != ""))])
    return out


@dataclass(frozen=True)
class ParticipantTable:
    rows: List[List[str]]
    dictionary: List[List[str]]
    quality_counts: Dict[str, int]


def build_participant_table(sources: CrfSources, mapping: Dict[str, str]) -> ParticipantTable:
    raw = load_raw(sources, list(mapping))
    data, counts = apply_quality_rules(raw)
    rows = derive_rows(data, mapping)
    return ParticipantTable(rows=rows, dictionary=participants_dictionary(rows), quality_counts=counts)

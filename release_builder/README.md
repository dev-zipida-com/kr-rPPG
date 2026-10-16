# krrppg_release: build and verify the KR-rPPG de-identified release

`krrppg_release` turns the custodian's source recordings into the KR-rPPG release (controlled access under
the DUA; manuscript Figure 2 / Table 5 / Table 6) and checks a built release. MIT licence (see the repository
root).

* **Build** (data custodian only): reads the source tree and the case report form (CRF) workbooks, never
  writes inside them, and writes
  * `KR-rPPG/pNNN/T0x/` with `video_bfdt_ffv1.avi` (BFDT de-identified, 8 x 8 blocks, keep ratio 0.02,
    i.e. only the block mean survives; FFV1, 640 x 480, 30 fps), `BVP_synchronized.csv`,
    `HR_synchronized.csv`, `SpO2_synchronized.csv`, `frames_timestamp.csv` (relative clock),
    `face_bboxes.csv`;
  * `KR-rPPG/metadata/` with the 24-column `participants.csv` (coded as manuscript Table 6: numbers for age,
    body measures, the nine clinical measurements and the comorbidity count, 0/1 or text for the others; see
    [Participant metadata](#participant-metadata)), its dictionary, `task_session_availability.csv` and
    `fold_assignment.csv`;
  * a **private** folder (`--private-dir`) with the random participant-ID mapping, per-session records and
    the build report. Never publish the private folder.
* **Verify** (anyone with the release): layout, metadata consistency, allowed values and rounding grids
  of the participant columns, per-session tables, absence of identifiers and dates, and that every video frame is block-constant
  with the expected frame count. With `--source` and `--mapping` (custodian) it also re-renders every file
  from the source and compares bytes.

## Install and test

```bash
pip install -r ../environment/requirements-release-builder.txt     # Python 3.9
python -m pytest                                                     # synthetic data only (about 1 min)
```

The tests build complete releases from a synthetic source tree (random-noise FFV1 videos, a session clock
starting at the round epoch 1,000,000,000 s, a placeholder recording date 2000-01-01, CRF rows drawn by a
seeded generator inside fixed ranges, participant numbers in the reserved synthetic range 900-999;
see `tests/conftest.py`). `tests/test_bfdt.py` compares the fast BFDT path with the independent reference
implementation in `../privacy/bfdt_transform.py`.

The permission tests give their temporary folders the caller's own group before setting a setgid bit, so
they also pass under a setgid temp parent; where the filesystem still refuses the bit they are skipped with
the reason. On network filesystems (NFS, SMB, ...) `test_tree_snapshot_sees_new_files_and_permission_changes`
is marked as a non-strict expected failure, because client-side attribute caching can hide a change made a
moment earlier; on local disks it is a normal test.

## Usage

```bash
# DUA users: check a downloaded release (read-only)
python -m krrppg_release verify --release-root /path/to/KR-rPPG --workers 8

# custodian: build (the source and the CRF are only read)
export KRRPPG_SOURCE=/path/to/source        # <site>/<group>/pNNN/T0x/ (site = A, B or C) + metadata/
export KRRPPG_PROTECTED_ROOTS=/path/to/source_dataset:/path/to/other_protected_folder
python -m krrppg_release build --output-dir /path/to/out --private-dir /path/to/private \
       --crf /path/to/crf.xlsx --crf-ild /path/to/ild.xlsx --workers 8
python -m krrppg_release verify --release-root /path/to/out/KR-rPPG --source "$KRRPPG_SOURCE" \
       --mapping /path/to/private/participant_id_map.csv --crf /path/to/crf.xlsx --crf-ild /path/to/ild.xlsx
```

Custodian paths and workbook settings come from the command line or the environment (built in are only the
three site letters A, B and C and the screening-number prefixes that map CRF rows to them, `clinical.SITE_PREFIX`):

| Variable | Meaning |
|---|---|
| `KRRPPG_SOURCE` | default for `--source` |
| `KRRPPG_PROTECTED_ROOTS` | extra folders the builder must never write to (`os.pathsep`-separated); the source is always protected |
| `KRRPPG_CRF_FILENAME` | CRF workbook name inside the source folder when `--crf` is not given (default `crf.xlsx`) |
| `KRRPPG_ILD_XLSX` | workbook with the ILD item when `--crf-ild` is not given |
| `KRRPPG_CRF_SHEET`, `KRRPPG_ILD_SHEET` | sheet names of the two workbooks (default: the first sheet of each) |

`--seed` exists for tests only; production builds draw the random release numbering once and keep it in
the private mapping. The default BFDT block size is 8 (the paper setting); `--block-size 16` (optional) uses
16 x 16 blocks with keep ratio 1/256, so that again only the block mean is kept; pass the same `--block-size`
to `verify`.

Group ownership is not built in: the builder never sets a Unix group. New files and folders get the group
of the output parent folder when that folder has the setgid bit, so choose it there, e.g.
`chgrp <your-group> /path/to/parent && chmod g+s /path/to/parent` (folders 0750 / files 0640 in the release,
0700 / 0600 in the private folder).

## Participant metadata

`KR-rPPG/metadata/participants.csv` holds the 24 variables of manuscript Table 6 with the same names and
order, one row per participant in release-ID order; a blank cell means missing (`ild`: not ascertained).
Since builder 1.4.0 every column is coded as manuscript Table 6 describes it (`clinical.CODINGS`); the
categories of builders 1.2.x-1.3.0 (age bands, height and weight bins, BMI classes, a comorbidity count
top-coded at `>=1`) are no longer used. The columns are of two kinds:

* **Categorical** (`clinical.CATEGORICAL_COLUMNS`): `site` as the collection-site letter `A`, `B` or `C`
  (since builder 1.5.0; the sites are not named), `sex` as recorded; `makeup`, `glasses`, `accessories`, `dm`,
  `htn`, `dyslipidemia` and `ild` as 0/1 (`ild` = interstitial lung disease, ascertained at site C only and blank
  elsewhere).
* **Numbers** (`clinical.NUMERIC_COLUMNS`), one note per column:

| Column | Coding | Written as |
|---|---|---|
| `age` | age at enrollment in years, as recorded in the CRF (a fractional age stops the build) | integer |
| `height` | rounded to 5 cm | integer |
| `weight` | rounded to 2 kg | even integer |
| `bmi` | weight / (height / 100)^2 of the released (rounded) height and weight, rounded to one decimal (`clinical.released_bmi`); blank when either is blank | one decimal |
| `sbp`, `dbp` | rounded to 5 mmHg | integer |
| `hr_rest` | CRF vital sign (not the benchmark reference heart rate), rounded to 2 bpm | even integer |
| `hba1c` | rounded to 0.5 % | one decimal |
| `fpg` | rounded to 10 mg/dL | integer |
| `creatinine` | rounded to 0.1 mg/dL | one decimal |
| `gfr` | CRF eGFR (CKD-EPI), rounded to 5 mL/min/1.73 m^2; computed with the CKD-EPI 2009 creatinine equation (no race coefficient) when the CRF value is missing | integer |
| `ast`, `alt` | rounded to 5 U/L | integer |
| `other_comorbidity_count` | number of positive items among cardiovascular disease, cerebrovascular disease, cancer, renal disease and lung disease (0-5, no top-coding); blank unless all five items are known; eGFR below 15 mL/min/1.73 m^2 counts as renal disease | integer |

Rounding (`clinical.round_to_step`) is numpy round half to even of round(value, 9) / step, times the step,
written with the column's decimals. Exact halves go to the even multiple (`height`: 157.5 -> 160 and
152.5 -> 150; `weight`: 71 -> 72 and 73 -> 72; `fpg`: 105 -> 100 and 115 -> 120). The 9-decimal snap comes
first, so float noise does not move a value across a half (17.499999999999996 -> 20 for step 5). The division
is binary floating point, so a decimal half that is stored just below the half goes down (`creatinine`:
0.95 -> 0.9). There is no top or bottom coding: values in either tail are only rounded. Unrounded heights,
weights and measurements are never written.

**Disclosure risk.** Only the categorical columns have categories; `verify` warns about any of them held by
fewer than 5 participants. The numeric columns, exact age in particular, can hold rare values, and
combinations of columns are not k-anonymous. The residual re-identification risk is managed under the data
use agreement (DUA) that governs access to the release.

All values are computed from the raw CRF values (not from the rounded `<source_root>/metadata/participants.csv`).
No data-quality check removes a value. The build report (`data_quality_rule_counts`) only counts the checks,
and the values are released as recorded in the CRF: height outside 120-210 cm, BMI of the CRF height and weight
outside 13-60 kg/m^2, SBP below 70 mmHg, DBP not below SBP, CRF eGFR and the CKD-EPI 2009 eGFR from creatinine
differing by more than 15 mL/min/1.73 m^2, AST at least 200 U/L with an AST/ALT ratio of at least 10, FPG below
40 mg/dL and a CRF heart rate outside 30-200 bpm (each `..._flagged_unchanged`; up to builder 1.3.0 all but
the heart-rate check blanked the value). Two in-memory corrections that add information are applied and counted:
eGFR computed with CKD-EPI 2009 when the CRF eGFR is missing and creatinine is recorded
(`egfr_computed_from_creatinine`), and eGFR below 15 counted as renal disease in `other_comorbidity_count`
(`renal_disease_recoded_from_egfr_below_15`).

`participants_dictionary.csv` has one row per column (group, the manuscript Table 6 description, unit or
coding, number of participants with a value); `other_comorbidity_count` gives its possible and observed
range (`0-5 possible (observed a-b)`). It is a pure function of `participants.csv` (`clinical.DICTIONARY`).

`verify` reports an error when the columns or their order differ; when a categorical cell is outside its
values; when a numeric cell is not plain decimal text with the column's decimals (no sign, exponent or
leading zero; whole numbers for `age` and `other_comorbidity_count`), is not a multiple of the step, or lies
outside `clinical.PLAUSIBLE_RANGES` (`age` 0-120, `height` 50-250, `weight` 10-300, `sbp` 50-300, `dbp`
20-200, `hr_rest` 20-250, `hba1c` 3.0-20.0, `fpg` 10-1000, `creatinine` 0.1-20.0, `gfr` 0-200, `ast` and `alt`
0-5000, `other_comorbidity_count` 0-5: wide sanity limits against unit or column mix-ups, not data-quality
rules); when `bmi` differs from `clinical.released_bmi` of the row's height and weight; or when the
dictionary, including the availability counts and the observed range, differs from the one re-derived from
`participants.csv`. Messages give the column, the count and the rule, never a cell value. It warns about
categories held by fewer than 5 participants (categorical columns only) and about release IDs that follow
site order.

Builders 1.3.0, 1.4.0 and 1.5.0 change only `participants.csv` and `participants_dictionary.csv`. Per-session outputs
are unchanged since 1.1.0 (`constants.SESSION_OUTPUT_VERSION`), so `build --metadata-only` rewrites the
metadata of an existing release without writing any session file.

## Limits

The build needs the original recordings and the CRF, so only the data custodian can run it. The
re-identification evaluation of the manuscript (Table 9) also needs the original videos and is not part
of this package.

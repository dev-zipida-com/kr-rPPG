"""Builds and verifies the KR-rPPG release (BFDT FFV1 videos, p001-p296 layout; controlled access under the DUA).

The source dataset (<source_root>) is only ever read; every write goes
through `safety.PathPolicy`, which refuses any path inside the source tree.
"""

__version__ = "1.4.0"  # 1.4.0: participants.csv coded as manuscript Table 6 in all 24 columns (age in years, height/weight rounded to 5 cm / 2 kg, BMI from the rounded values, comorbidity count 0-5; no categories), data-quality checks no longer blank values (counted only), CKD-EPI 2009 fill and kidney-failure renal recode kept; 1.3.0: nine clinical measurements released as rounded numbers, the other columns keep their categories, FPG below 40 mg/dL blanked
# Session outputs are unchanged since 1.1.0 (constants.SESSION_OUTPUT_VERSION), so 1.1 session records stay valid.

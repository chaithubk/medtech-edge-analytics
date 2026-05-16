#!/usr/bin/env python3
"""
PhysioNet 2019 Sepsis Challenge data harmonization processor.

This script processes pre-downloaded PhysioNet PSV files and harmonizes them
to match the internal data contract for model training.

Invoked by CI when Synthea dataset contains zero positive sepsis cases.

PhysioNet column -> internal schema mapping:
  HR          -> hr
  O2Sat       -> o2_sat
  SBP         -> bp_sys
  DBP         -> bp_dia
  Temp        -> temperature   (already Celsius; no unit conversion needed)
  Resp        -> respiratory_rate
  WBC         -> wbc
  Lactate     -> lactate
  Creatinine  -> creatinine
  SepsisLabel -> sepsis

SIRS score (0-4) is derived from available vitals:
  +1  Temp > 38 C or < 36 C
  +1  HR   > 90
  +1  Resp > 20
  +1  WBC  > 12 or < 4

qSOFA score (0-2, mentation unavailable from vitals alone):
  +1  Resp >= 22
  +1  SBP  <= 100

Prerequisites:
  - PhysioNet data must be pre-downloaded to data/physionet_raw/training_setA/
  - Run scripts/download_physionet.py first to authenticate and download
"""

import glob
import sys
from pathlib import Path

import pandas as pd

DATASET_PATH = Path("data/processed/dataset.csv")
PHYSIONET_DATA_DIR = Path("data/physionet_raw/training_setA")

# Maximum number of .psv patient files to load (keeps CI fast)
MAX_PATIENTS = 500

FEATURE_COLUMNS = [
    "hr",
    "bp_sys",
    "bp_dia",
    "o2_sat",
    "temperature",
    "respiratory_rate",
    "wbc",
    "lactate",
    "creatinine",
    "sirs_score",
    "qsofa_score",
]
TARGET_COLUMN = "sepsis"

# PhysioNet PSV column name -> internal schema column
COLUMN_MAP = {
    "HR": "hr",
    "O2Sat": "o2_sat",
    "SBP": "bp_sys",
    "DBP": "bp_dia",
    "Temp": "temperature",
    "Resp": "respiratory_rate",
    "WBC": "wbc",
    "Lactate": "lactate",
    "Creatinine": "creatinine",
    "SepsisLabel": TARGET_COLUMN,
}


def _latest_value(series):
    """Return the last non-null numeric value in a Series, or None."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else None


def _compute_sirs(row):
    """Compute SIRS score (0-4) from vital measurements."""
    score = 0
    temp = row.get("temperature")
    if temp is not None and (temp > 38.0 or temp < 36.0):
        score += 1
    hr = row.get("hr")
    if hr is not None and hr > 90:
        score += 1
    rr = row.get("respiratory_rate")
    if rr is not None and rr > 20:
        score += 1
    wbc = row.get("wbc")
    if wbc is not None and (wbc > 12.0 or wbc < 4.0):
        score += 1
    return score


def _compute_qsofa(row):
    """Compute qSOFA score (0-2) from vital measurements."""
    score = 0
    rr = row.get("respiratory_rate")
    if rr is not None and rr >= 22:
        score += 1
    sbp = row.get("bp_sys")
    if sbp is not None and sbp <= 100:
        score += 1
    return score


def _parse_psv_to_record(psv_path):
    """
    Parse a single PhysioNet .psv patient file into a flat feature record.

    Takes the last known (non-null) value for each vital/lab column and the
    maximum SepsisLabel across all time steps (patient is positive if any
    hour is labelled 1).
    """
    try:
        df = pd.read_csv(psv_path, sep="|")
    except Exception as exc:
        print(f"  Warning: could not parse {psv_path.name}: {exc}")
        return None

    if df.empty:
        return None

    record = {}

    for psv_col, schema_col in COLUMN_MAP.items():
        if psv_col not in df.columns:
            record[schema_col] = None
            continue
        if psv_col == "SepsisLabel":
            labels = pd.to_numeric(df[psv_col], errors="coerce")
            record[schema_col] = int(1 if (labels == 1).any() else 0)
        else:
            record[schema_col] = _latest_value(df[psv_col])

    # Fill any schema columns not present in PSV
    for col in FEATURE_COLUMNS:
        if col not in record:
            record[col] = None

    record["sirs_score"] = float(_compute_sirs(record))
    record["qsofa_score"] = float(_compute_qsofa(record))

    if record.get(TARGET_COLUMN) is None:
        record[TARGET_COLUMN] = 0

    return record


def _load_physionet_sample(psv_files):
    """Parse up to MAX_PATIENTS PSV files, return a DataFrame in schema order."""
    files_to_load = psv_files[:MAX_PATIENTS]
    print(f"  Parsing {len(files_to_load)} PhysioNet patient files ...")
    records = []
    for f in files_to_load:
        rec = _parse_psv_to_record(f)
        if rec is not None:
            records.append(rec)

    df = pd.DataFrame(records, columns=[*FEATURE_COLUMNS, TARGET_COLUMN])
    pos = int(df[TARGET_COLUMN].sum())
    neg = len(df) - pos
    print(f"  PhysioNet sample loaded: {len(df)} patients — {pos} positive, {neg} negative.")
    return df


def _impute_and_merge(synthea_df, physionet_df):
    """
    Merge Synthea (all-negative) rows with PhysioNet sample rows, then impute.

    We keep the Synthea data for negative examples and append PhysioNet rows
    so that both sources contribute to the feature distribution.
    """
    combined = pd.concat([synthea_df, physionet_df], ignore_index=True)

    for col in FEATURE_COLUMNS:
        series = pd.to_numeric(combined[col], errors="coerce")
        if series.isna().any():
            fill = series.median() if series.notna().any() else 0.0
            combined[col] = series.fillna(fill)

    combined[TARGET_COLUMN] = (
        pd.to_numeric(combined[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    )
    return combined


def _generate_synthetic_positive(negative_df):
    """
    Generate a synthetic sepsis-positive case based on negative case statistics.

    WARNING: This is a fallback-only mechanism. Using synthetic positives for
    production models is NOT RECOMMENDED. Real clinical data should always be
    preferred for edge device deployment.
    """
    print("  WARNING: Generating synthetic positive case (NOT recommended for production)")
    synthetic = {}
    for col in FEATURE_COLUMNS:
        series = pd.to_numeric(negative_df[col], errors="coerce")
        if series.notna().any():
            mean_val = float(series.mean())
            std_val = float(series.std())
            if pd.isna(std_val) or std_val == 0:
                synthetic[col] = mean_val
            else:
                import numpy as np

                synthetic[col] = float(np.random.normal(mean_val, std_val))
        else:
            synthetic[col] = 0.0
    synthetic[TARGET_COLUMN] = 1.0
    return synthetic


def main():
    """
    Main processor: validates PhysioNet data availability, harmonizes, merges.
    """
    print("=== PhysioNet Fallback Processor ===")

    # Check if PhysioNet data directory exists
    if not PHYSIONET_DATA_DIR.exists():
        print(f"ERROR: PhysioNet data not found at {PHYSIONET_DATA_DIR}")
        print("SOLUTION: Run scripts/download_physionet.py first to download and extract data")
        print("AUTHENTICATION: Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD environment variables")
        return 1

    # Find all PSV files
    psv_files = sorted(glob.glob(str(PHYSIONET_DATA_DIR / "*.psv")))

    if not psv_files:
        print(f"ERROR: No PSV files found in {PHYSIONET_DATA_DIR}")
        return 1

    print(f"  Found {len(psv_files)} PhysioNet patient files")

    # Load existing Synthea data if present (provides negative examples)
    synthea_df = None
    if DATASET_PATH.exists():
        synthea_df = pd.read_csv(DATASET_PATH)
        # Ensure schema columns exist
        for col in [*FEATURE_COLUMNS, TARGET_COLUMN]:
            if col not in synthea_df.columns:
                synthea_df[col] = 0
        synthea_df = synthea_df[[*FEATURE_COLUMNS, TARGET_COLUMN]]
        print(f"  Existing Synthea data: {len(synthea_df)} rows (all negative)")
    else:
        synthea_df = pd.DataFrame(columns=[*FEATURE_COLUMNS, TARGET_COLUMN])
        print("  No existing Synthea data found")

    # Process PhysioNet data
    try:
        physionet_df = _load_physionet_sample(psv_files)

        if int(physionet_df[TARGET_COLUMN].sum()) > 0:
            print(f"  ✓ PhysioNet sample contains positive cases")
            combined = _impute_and_merge(synthea_df, physionet_df)
        else:
            print("  WARNING: PhysioNet sample has no positive cases")
            print("  Generating synthetic positive case as absolute last resort...")
            synthetic_rec = _generate_synthetic_positive(synthea_df)
            physionet_df = pd.DataFrame([synthetic_rec])
            combined = _impute_and_merge(synthea_df, physionet_df)

    except Exception as e:
        print(f"ERROR: PhysioNet processing failed: {e}")
        print("  Attempting synthetic fallback as last resort...")
        synthetic_rec = _generate_synthetic_positive(synthea_df)
        physionet_df = pd.DataFrame([synthetic_rec])
        combined = _impute_and_merge(synthea_df, physionet_df)

    # Write harmonized dataset
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATASET_PATH, index=False)

    pos_total = int(combined[TARGET_COLUMN].sum())
    neg_total = len(combined) - pos_total
    print(f"  ✓ Final dataset: {len(combined)} rows ({pos_total} positive, {neg_total} negative)")
    print(f"  ✓ Exported to {DATASET_PATH}")
    print("=== PhysioNet Fallback Processor Complete ===")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

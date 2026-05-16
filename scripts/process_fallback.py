#!/usr/bin/env python3
"""
PhysioNet 2019 Sepsis Challenge fallback processor.

This script is invoked automatically by the CI pipeline when the processed
Synthea dataset contains zero positive sepsis cases (SepsisLabel == 1).

Behaviour:
  1. Reads data/processed/dataset.csv and checks for positive cases.
  2. If positives are present, exits immediately (no-op).
  3. If no positives are found, downloads a micro-sample of the PhysioNet
     Challenge 2019 Training Set A, parses the pipe-separated (.psv) files,
     maps columns to our standardised feature schema, computes SIRS and qSOFA
     scores, and overwrites data/processed/dataset.csv so that downstream
     training always receives a balanced dataset.

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
"""

import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path("data/processed/dataset.csv")
PHYSIONET_RAW_DIR = Path("data/physionet_raw")

PHYSIONET_ZIP_URL = (
    "https://physionet.org/static/published-projects/challenge-2019/"
    "physionet-challenge-2019-training-set-a.zip"
)

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


def _has_positives(csv_path: Path) -> bool:
    """Return True if csv_path exists and contains at least one sepsis==1 row."""
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
        return bool(df[TARGET_COLUMN].sum() > 0)
    except Exception:
        return False


def _latest_value(series: pd.Series) -> Optional[float]:
    """Return the last non-null numeric value in a Series, or None."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else None


def _compute_sirs(row: Dict[str, Optional[float]]) -> int:
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


def _compute_qsofa(row: Dict[str, Optional[float]]) -> int:
    score = 0
    rr = row.get("respiratory_rate")
    if rr is not None and rr >= 22:
        score += 1
    sbp = row.get("bp_sys")
    if sbp is not None and sbp <= 100:
        score += 1
    return score


def _download_physionet_zip() -> Path:
    """Download PhysioNet Training Set A zip, return local path."""
    PHYSIONET_RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = PHYSIONET_RAW_DIR / "physionet-challenge-2019-training-set-a.zip"
    if zip_path.exists():
        print(f"  PhysioNet zip already present at {zip_path}, skipping download.")
        return zip_path
    print(f"  Downloading PhysioNet 2019 Set A from physionet.org ...")
    urllib.request.urlretrieve(PHYSIONET_ZIP_URL, zip_path)  # noqa: S310
    print(f"  Download complete: {zip_path}")
    return zip_path


def _extract_psv_files(zip_path: Path) -> List[Path]:
    """Extract .psv files from the zip into PHYSIONET_RAW_DIR/psv/, return list."""
    psv_dir = PHYSIONET_RAW_DIR / "psv"
    psv_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(psv_dir.glob("*.psv"))
    if existing:
        print(f"  {len(existing)} .psv files already extracted, skipping unzip.")
        return existing
    print("  Extracting .psv patient files ...")
    with zipfile.ZipFile(zip_path) as zf:
        psv_members = [m for m in zf.namelist() if m.endswith(".psv")]
        for member in psv_members:
            filename = Path(member).name
            data = zf.read(member)
            (psv_dir / filename).write_bytes(data)
    extracted = sorted(psv_dir.glob("*.psv"))
    print(f"  Extracted {len(extracted)} patient files.")
    return extracted


def _parse_psv_to_record(psv_path: Path) -> Optional[Dict]:
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

    record: Dict[str, Optional[float]] = {}

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


def _load_physionet_sample(psv_files: List[Path]) -> pd.DataFrame:
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


def _impute_and_merge(synthea_df: pd.DataFrame, physionet_df: pd.DataFrame) -> pd.DataFrame:
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


def main() -> None:
    print("=== PhysioNet Fallback Check ===")

    if _has_positives(DATASET_PATH):
        pos = int(pd.read_csv(DATASET_PATH)[TARGET_COLUMN].sum())
        print(f"  dataset.csv has {pos} positive sepsis case(s). Fallback not required.")
        return

    print(
        "::warning:: No sepsis cases found in Synthea cohort. Initiating PhysioNet 2019 fallback download loop."
    )

    # Load existing Synthea data if present (provides negative examples)
    synthea_df: pd.DataFrame
    if DATASET_PATH.exists():
        synthea_df = pd.read_csv(DATASET_PATH)
        # Ensure schema columns exist
        for col in [*FEATURE_COLUMNS, TARGET_COLUMN]:
            if col not in synthea_df.columns:
                synthea_df[col] = 0
        synthea_df = synthea_df[[*FEATURE_COLUMNS, TARGET_COLUMN]]
        print(f"  Existing Synthea data: {len(synthea_df)} rows (all negative).")
    else:
        synthea_df = pd.DataFrame(columns=[*FEATURE_COLUMNS, TARGET_COLUMN])

    zip_path = _download_physionet_zip()
    psv_files = _extract_psv_files(zip_path)

    if not psv_files:
        print("::error:: No .psv files found in PhysioNet zip. Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    physionet_df = _load_physionet_sample(psv_files)

    if int(physionet_df[TARGET_COLUMN].sum()) == 0:
        print(
            "  Warning: PhysioNet sample also yielded no positive cases. Training will likely fail.",
            file=sys.stderr,
        )

    combined = _impute_and_merge(synthea_df, physionet_df)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATASET_PATH, index=False)

    pos_total = int(combined[TARGET_COLUMN].sum())
    print(
        f"  Final dataset: {len(combined)} rows — {pos_total} positive, {len(combined) - pos_total} negative."
    )
    print(f"  Written to {DATASET_PATH}")
    print("=== PhysioNet Fallback Complete ===")


if __name__ == "__main__":
    main()

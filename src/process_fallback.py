#!/usr/bin/env python3
"""
PhysioNet Sample Data Processor

Reads committed sample PhysioNet PSV files and harmonizes them to the
internal data contract for model training. This ensures the model always
trains with realistic sepsis and non-sepsis cases.

Sample data location: data/physionet_sample/training_setA/
- 5 healthy patient records (sepsis = 0)
- 5 sepsis patient records (sepsis = 1)
- Format: Pipe-separated values (PSV) with realistic vital signs
- Generated via: scripts/generate_sample_data.py

Output: data/processed/dataset.csv (merged with Synthea if available)
"""

import sys
from pathlib import Path

import pandas as pd


def compute_sirs(row):
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


def compute_qsofa(row):
    """Compute qSOFA score (0-2) from vital measurements."""
    score = 0
    rr = row.get("respiratory_rate")
    if rr is not None and rr >= 22:
        score += 1
    sbp = row.get("bp_sys")
    if sbp is not None and sbp <= 100:
        score += 1
    return score


def process_sample_data():
    """Read and harmonize committed PhysioNet sample data."""

    sample_dir = Path("data/physionet_sample/training_setA")

    if not sample_dir.exists():
        print(f"ERROR: Sample data directory not found: {sample_dir}")
        print("Run: python scripts/generate_sample_data.py")
        return None

    # Find all PSV files
    psv_files = sorted(sample_dir.glob("*.psv"))

    if not psv_files:
        print(f"ERROR: No PSV files found in {sample_dir}")
        return None

    print(f"Processing {len(psv_files)} sample patient records...")

    records = []
    sepsis_positive = 0
    sepsis_negative = 0

    for filepath in psv_files:
        try:
            # Read PSV file (pipe-separated)
            df = pd.read_csv(filepath, sep="|")

            if df.empty:
                continue

            # Extract median values from time series
            record = {
                "hr": df["HR"].median(),
                "bp_sys": df["SBP"].median(),
                "bp_dia": df["DBP"].median(),
                "o2_sat": df["O2Sat"].median(),
                "temperature": df["Temp"].median(),
                "respiratory_rate": df["Resp"].median(),
                "wbc": df["WBC"].median(),
                "lactate": df["Lactate"].median(),
                "creatinine": df["Creatinine"].median(),
            }

            # Check for sepsis (if ANY SepsisLabel==1 in time series, patient is positive)
            has_sepsis = 1 if (df["SepsisLabel"] == 1).any() else 0
            record["sepsis"] = has_sepsis

            # Compute clinical scores
            record["sirs_score"] = compute_sirs(record)
            record["qsofa_score"] = compute_qsofa(record)

            records.append(record)

            if has_sepsis:
                sepsis_positive += 1
            else:
                sepsis_negative += 1

        except Exception as e:
            print(f"  Warning: Failed to parse {filepath.name}: {e}")
            continue

    if not records:
        print("ERROR: No valid records extracted from sample data")
        return None

    df_sample = pd.DataFrame(records)

    print(f"✓ Extracted {len(records)} patient records")
    print(f"  Sepsis positive: {sepsis_positive}")
    print(f"  Sepsis negative: {sepsis_negative}")
    print(f"  Prevalence: {sepsis_positive/len(records)*100:.1f}%")

    return df_sample


def merge_with_synthea(df_sample):
    """Merge sample data with existing Synthea data if available."""

    synthea_path = Path("data/processed/dataset.csv")

    if not synthea_path.exists():
        print("No existing Synthea data found. Using sample data only.")
        return df_sample

    try:
        df_synthea = pd.read_csv(synthea_path)
        print(f"Merging with existing Synthea data ({len(df_synthea)} records)...")

        # Ensure same columns
        for col in df_sample.columns:
            if col not in df_synthea.columns:
                df_synthea[col] = 0

        df_synthea = df_synthea[df_sample.columns]

        # Concatenate
        df_merged = pd.concat([df_synthea, df_sample], ignore_index=True)
        print(f"✓ Merged dataset: {len(df_merged)} total records")

        return df_merged

    except Exception as e:
        print(f"Warning: Could not merge Synthea data: {e}")
        return df_sample


def main():
    """Main processor."""

    print("=" * 70)
    print("PhysioNet Sample Data Processor")
    print("=" * 70)
    print()

    # Process sample data
    df_sample = process_sample_data()
    if df_sample is None:
        return False

    print()

    # Merge with Synthea if available
    df_final = merge_with_synthea(df_sample)

    # Handle missing values
    print("Applying missing value imputation...")
    numeric_cols = df_final.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df_final[col].isna().any():
            df_final[col].fillna(df_final[col].median(), inplace=True)

    # Export
    output_path = Path("data/processed/dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    print(f"✓ Dataset exported to {output_path}")
    print()

    # Final statistics
    positive_count = int((df_final["sepsis"] == 1).sum())
    negative_count = len(df_final) - positive_count

    print("Final Dataset Statistics:")
    print(f"  Total patients: {len(df_final)}")
    print(f"  Sepsis positive: {positive_count}")
    print(f"  Sepsis negative: {negative_count}")
    print(f"  Prevalence: {positive_count/len(df_final)*100:.1f}%")
    print()
    print("=" * 70)
    print("Ready for model training!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

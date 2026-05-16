#!/usr/bin/env python3
"""
PhysioNet Challenge 2019 data loader for sepsis detection.

Loads CSV data from PhysioNet Challenge 2019, extracts sepsis labels and vitals,
maps to feature schema, and outputs processed dataset matching flatten_fhir.py format.

Download from: https://physionet.org/content/challenge-2019/1.0.0/
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Feature mapping: PhysioNet columns to our schema
PHYSIONET_FEATURE_MAP = {
    "HR": "hr",
    "O2Sat": "o2_sat",
    "SBP": "bp_sys",
    "DBP": "bp_dia",
    "Temp": "temperature",
    "RR": "respiratory_rate",
    "WBC": "wbc",
    "Lactate": "lactate",
    "Creatinine": "creatinine",
}

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

PHYSIONET_DIR = Path("data/raw_physionet")
PROCESSED_DIR = Path("data/processed")


def load_physionet_csvs() -> List[Tuple[str, pd.DataFrame]]:
    """
    Load all PhysioNet Challenge 2019 CSV files from raw_physionet directory.

    Returns:
        List of (patient_id, dataframe) tuples
    """
    if not PHYSIONET_DIR.exists():
        raise FileNotFoundError(
            f"PhysioNet directory not found: {PHYSIONET_DIR}. "
            "Download from https://physionet.org/content/challenge-2019/1.0.0/"
        )

    patient_data = []
    csv_files = list(PHYSIONET_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {PHYSIONET_DIR}. "
            "Ensure you've downloaded and extracted PhysioNet Challenge 2019 data."
        )

    for csv_file in csv_files:
        try:
            patient_id = csv_file.stem
            df = pd.read_csv(csv_file)
            patient_data.append((patient_id, df))
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")

    return patient_data


def extract_physionet_features(patient_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Extract latest vital signs and derived scores from PhysioNet patient data.

    PhysioNet provides time-series data. We extract the latest non-null values.
    """
    features: Dict[str, Optional[float]] = {col: None for col in FEATURE_COLUMNS}

    # Map PhysioNet columns to our schema and extract latest values
    for physionet_col, feature_col in PHYSIONET_FEATURE_MAP.items():
        if physionet_col in patient_df.columns:
            # Get latest non-null value
            valid_values = pd.to_numeric(patient_df[physionet_col], errors="coerce").dropna()
            if not valid_values.empty:
                features[feature_col] = float(valid_values.iloc[-1])

    # PhysioNet may have SIRS and qSOFA scores; extract if present
    if "SIRS" in patient_df.columns:
        val = pd.to_numeric(patient_df["SIRS"], errors="coerce").dropna()
        if not val.empty:
            features["sirs_score"] = float(val.iloc[-1])

    if "qSOFA" in patient_df.columns:
        val = pd.to_numeric(patient_df["qSOFA"], errors="coerce").dropna()
        if not val.empty:
            features["qsofa_score"] = float(val.iloc[-1])

    return features


def extract_sepsis_label_physionet(patient_df: pd.DataFrame) -> int:
    """
    Extract sepsis label from PhysioNet data.

    PhysioNet Challenge 2019 has a 'SepsisLabel' column:
    - 1 = sepsis occurred
    - 0 = no sepsis
    """
    if "SepsisLabel" not in patient_df.columns:
        print("Warning: 'SepsisLabel' column not found in patient data")
        return 0

    # SepsisLabel is time-series (0/1 for each hour)
    # Patient has sepsis if ANY time point is labeled 1
    labels = pd.to_numeric(patient_df["SepsisLabel"], errors="coerce")
    return 1 if (labels == 1).any() else 0


def load_physionet_to_dataframe() -> Tuple[pd.DataFrame, List[Tuple[str, int]]]:
    """
    Load all PhysioNet CSVs and flatten to DataFrame matching flatten_fhir.py format.

    Returns:
        (DataFrame with features and sepsis label, list of (patient_id, sepsis_label))
    """
    patient_data_list = load_physionet_csvs()
    records = []
    patient_sepsis_labels = []

    for patient_id, patient_df in patient_data_list:
        features = extract_physionet_features(patient_df)
        sepsis_label = extract_sepsis_label_physionet(patient_df)

        patient_sepsis_labels.append((patient_id, sepsis_label))

        record = {col: features.get(col) for col in FEATURE_COLUMNS}
        record[TARGET_COLUMN] = sepsis_label
        records.append(record)

    df = pd.DataFrame.from_records(records, columns=[*FEATURE_COLUMNS, TARGET_COLUMN])
    return df, patient_sepsis_labels


def apply_imputation_and_scaling(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply robust multi-stage imputation and StandardScaler to dataset.
    Matches flatten_fhir.py logic.
    """
    if df.empty:
        return df.copy(), StandardScaler()

    df_imputed = df.copy()
    for col in FEATURE_COLUMNS:
        series = pd.to_numeric(df_imputed[col], errors="coerce")

        if series.isna().sum() == 0:
            df_imputed[col] = series
            continue

        if series.notna().any():
            median_val = series.median()
            if pd.isna(median_val):
                fill_value = float(series.mean())
            else:
                fill_value = float(median_val)
        else:
            fill_value = 0.0

        df_imputed[col] = series.fillna(fill_value)

    scaler = StandardScaler()
    df_imputed[FEATURE_COLUMNS] = scaler.fit_transform(df_imputed[FEATURE_COLUMNS])
    return df_imputed, scaler


def main():
    """Main PhysioNet data loading pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PhysioNet Challenge 2019 CSV files...")
    df, patient_sepsis_labels = load_physionet_to_dataframe()
    print(f"Loaded {len(df)} patient records")

    # Debug: Show which patients have sepsis
    sepsis_patients = [pid for pid, label in patient_sepsis_labels if label == 1]
    if sepsis_patients:
        print(f"\n✓ Found {len(sepsis_patients)} patient(s) with sepsis:")
        for pid in sepsis_patients[:10]:
            print(f"  - {pid}")
        if len(sepsis_patients) > 10:
            print(f"  ... and {len(sepsis_patients) - 10} more")
    else:
        print("\n✗ WARNING: No patients with sepsis found!")

    print("Applying imputation and scaling...")
    df_processed, scaler = apply_imputation_and_scaling(df)

    dataset_path = PROCESSED_DIR / "dataset.csv"
    df_processed.to_csv(dataset_path, index=False)
    print(f"Saved processed dataset to {dataset_path}")

    scaler_path = PROCESSED_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved fitted scaler to {scaler_path}")

    print("\nDataset statistics:")
    print(f"  Sepsis cases: {df_processed['sepsis'].sum()}")
    print(f"  Healthy cases: {len(df_processed) - df_processed['sepsis'].sum()}")
    print(f"  Class balance: {df_processed['sepsis'].mean():.2%} positive")


if __name__ == "__main__":
    main()

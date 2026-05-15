#!/usr/bin/env python3
"""
FHIR data flattening engine for Synthea sepsis dataset.

Extracts sepsis diagnoses (SNOMED 91302003) and vital LOINC codes,
applies StandardScaler, and exports preprocessed dataset with scaler artifact.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

LOINC_CODES = {
    "8867-4": "heart_rate",
    "8310-5": "body_temperature",
    "8480-6": "systolic_bp",
    "6690-2": "wbc",
}

SEPSIS_SNOMED_CODE = "91302003"
RAW_FHIR_DIR = Path("data/raw_fhir")
PROCESSED_DIR = Path("data/processed")


def extract_sepsis_label(patient_data: Dict) -> int:
    """
    Extract sepsis diagnosis from patient FHIR bundle.

    Args:
        patient_data: Parsed FHIR bundle for a patient

    Returns:
        1 if patient has sepsis diagnosis, 0 otherwise
    """
    if "entry" not in patient_data:
        return 0

    for entry in patient_data["entry"]:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Condition":
            coding_list = resource.get("code", {}).get("coding", [])
            for coding in coding_list:
                if coding.get("code") == SEPSIS_SNOMED_CODE:
                    return 1

    return 0


def extract_vital_signs(patient_data: Dict) -> Dict[str, Optional[float]]:
    """
    Extract vital signs observations from FHIR bundle.

    Args:
        patient_data: Parsed FHIR bundle for a patient

    Returns:
        Dictionary mapping vital names to values (or None if missing)
    """
    vitals: Dict[str, Optional[float]] = {vital: None for vital in LOINC_CODES.values()}

    if "entry" not in patient_data:
        return vitals

    for entry in patient_data["entry"]:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Observation":
            coding_list = resource.get("code", {}).get("coding", [])
            for coding in coding_list:
                loinc_code = coding.get("code")
                if loinc_code in LOINC_CODES:
                    vital_name = LOINC_CODES[loinc_code]
                    value_quantity = resource.get("value", {})
                    if isinstance(value_quantity, dict) and "value" in value_quantity:
                        vitals[vital_name] = float(value_quantity["value"])

    return vitals


def load_fhir_bundles() -> List[Dict]:
    """
    Load all FHIR JSON bundles from raw directory.

    Returns:
        List of parsed FHIR bundles
    """
    bundles = []

    if not RAW_FHIR_DIR.exists():
        raise FileNotFoundError(f"FHIR directory not found: {RAW_FHIR_DIR}")

    for json_file in RAW_FHIR_DIR.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                bundle = json.load(f)
                if bundle.get("resourceType") == "Bundle":
                    bundles.append(bundle)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return bundles


def flatten_fhir_to_dataframe(bundles: List[Dict]) -> pd.DataFrame:
    """
    Flatten FHIR bundles into structured DataFrame.

    Args:
        bundles: List of FHIR bundles

    Returns:
        DataFrame with features and target label
    """
    records = []

    for bundle in bundles:
        vitals = extract_vital_signs(bundle)
        sepsis_label = extract_sepsis_label(bundle)

        record = {
            "heart_rate": vitals["heart_rate"],
            "body_temperature": vitals["body_temperature"],
            "systolic_bp": vitals["systolic_bp"],
            "wbc": vitals["wbc"],
            "sepsis": sepsis_label,
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


def apply_imputation_and_scaling(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply robust imputation and StandardScaler to dataset.

    Args:
        df: Raw DataFrame with potential missing values

    Returns:
        Tuple of (imputed and scaled DataFrame, fitted scaler)
    """
    feature_cols = ["heart_rate", "body_temperature", "systolic_bp", "wbc"]

    imputer = SimpleImputer(strategy="median")
    df_imputed = df.copy()
    df_imputed[feature_cols] = imputer.fit_transform(df[feature_cols])

    scaler = StandardScaler()
    df_imputed[feature_cols] = scaler.fit_transform(df_imputed[feature_cols])

    return df_imputed, scaler


def main():
    """Main pipeline execution."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FHIR bundles...")
    bundles = load_fhir_bundles()
    print(f"Loaded {len(bundles)} bundles")

    print("Flattening FHIR data...")
    df = flatten_fhir_to_dataframe(bundles)
    print(f"Created dataset with {len(df)} records")

    print("Applying imputation and scaling...")
    df_processed, scaler = apply_imputation_and_scaling(df)

    dataset_path = PROCESSED_DIR / "dataset.csv"
    df_processed.to_csv(dataset_path, index=False)
    print(f"Saved processed dataset to {dataset_path}")

    scaler_path = PROCESSED_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved fitted scaler to {scaler_path}")

    print("Dataset statistics:")
    print(f"  Sepsis cases: {df_processed['sepsis'].sum()}")
    print(f"  Healthy cases: {len(df_processed) - df_processed['sepsis'].sum()}")
    print(f"  Class balance: {df_processed['sepsis'].mean():.2%} positive")


if __name__ == "__main__":
    main()

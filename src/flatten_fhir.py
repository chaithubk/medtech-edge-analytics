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

FEATURE_COLUMNS = ["heart_rate", "body_temperature", "systolic_bp", "wbc"]
TARGET_COLUMN = "sepsis"

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
            # Most FHIR Observation values are under valueQuantity.value.
            value_quantity = resource.get("valueQuantity", {})
            coding_list = resource.get("code", {}).get("coding", [])
            for coding in coding_list:
                loinc_code = coding.get("code")
                if loinc_code in LOINC_CODES:
                    vital_name = LOINC_CODES[loinc_code]
                    if isinstance(value_quantity, dict) and "value" in value_quantity:
                        vitals[vital_name] = float(value_quantity["value"])

            # Blood pressure often arrives as a panel Observation with components.
            for component in resource.get("component", []):
                comp_coding_list = component.get("code", {}).get("coding", [])
                comp_value_quantity = component.get("valueQuantity", {})
                if not (isinstance(comp_value_quantity, dict) and "value" in comp_value_quantity):
                    continue
                for comp_coding in comp_coding_list:
                    loinc_code = comp_coding.get("code")
                    if loinc_code in LOINC_CODES:
                        vital_name = LOINC_CODES[loinc_code]
                        vitals[vital_name] = float(comp_value_quantity["value"])

    return vitals


def load_fhir_bundles() -> List[Dict]:
    """
    Load all FHIR JSON bundles from raw directory.

    Returns:
        List of parsed FHIR bundles
    """
    bundles: List[Dict] = []
    resources: List[Dict] = []

    if not RAW_FHIR_DIR.exists():
        raise FileNotFoundError(f"FHIR directory not found: {RAW_FHIR_DIR}")

    files = list(RAW_FHIR_DIR.rglob("*.json")) + list(RAW_FHIR_DIR.rglob("*.ndjson"))

    for json_file in files:
        try:
            with open(json_file, "r") as f:
                if json_file.suffix == ".ndjson":
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if item.get("resourceType") == "Bundle":
                            bundles.append(item)
                        elif item.get("resourceType"):
                            resources.append(item)
                    continue

                parsed = json.load(f)
                parsed_items = parsed if isinstance(parsed, list) else [parsed]
                for item in parsed_items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("resourceType") == "Bundle":
                        bundles.append(item)
                    elif item.get("resourceType"):
                        resources.append(item)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    if bundles:
        return bundles

    # Fallback: if export produced standalone FHIR resources, group by patient.
    if resources:
        grouped: Dict[str, List[Dict]] = {}

        def patient_key(resource: Dict) -> str:
            if resource.get("resourceType") == "Patient":
                rid = resource.get("id")
                return str(rid) if rid else "unknown"

            ref = (
                resource.get("subject", {}).get("reference")
                or resource.get("patient", {}).get("reference")
                or ""
            )
            if isinstance(ref, str) and ref:
                return ref.split("/")[-1]
            return "unknown"

        for resource in resources:
            key = patient_key(resource)
            grouped.setdefault(key, []).append(resource)

        for patient_resources in grouped.values():
            bundles.append(
                {
                    "resourceType": "Bundle",
                    "entry": [{"resource": r} for r in patient_resources],
                }
            )

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

    df = pd.DataFrame.from_records(records, columns=[*FEATURE_COLUMNS, TARGET_COLUMN])
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
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    if df.empty:
        return df.copy(), StandardScaler()

    imputer = SimpleImputer(strategy="median")
    df_imputed = df.copy()
    df_imputed[FEATURE_COLUMNS] = imputer.fit_transform(df[FEATURE_COLUMNS])

    scaler = StandardScaler()
    df_imputed[FEATURE_COLUMNS] = scaler.fit_transform(df_imputed[FEATURE_COLUMNS])

    return df_imputed, scaler


def main():
    """Main pipeline execution."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FHIR bundles...")
    bundles = load_fhir_bundles()
    print(f"Loaded {len(bundles)} bundles")

    if not bundles:
        raise ValueError(
            f"No FHIR bundles found under {RAW_FHIR_DIR}. "
            "Ensure Synthea output contains Bundle JSON files."
        )

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

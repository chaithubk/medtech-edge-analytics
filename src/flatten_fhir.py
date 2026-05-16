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
from sklearn.preprocessing import StandardScaler

# Expanded LOINC codes and feature columns to match schema
LOINC_CODES = {
    "8867-4": "hr",  # Heart rate
    "8480-6": "bp_sys",  # Systolic BP
    "8462-4": "bp_dia",  # Diastolic BP
    "59408-5": "o2_sat",  # O2 saturation
    "8310-5": "temperature",  # Body temperature
    "9279-1": "respiratory_rate",  # Respiratory rate
    "6690-2": "wbc",  # WBC
    "2524-7": "lactate",  # Lactate
    "2160-0": "creatinine",  # Creatinine
    # sirs_score, qsofa_score are derived, not LOINC, but included as features
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
    Extract all relevant vital signs and labs from FHIR bundle.
    """
    vitals: Dict[str, Optional[float]] = {col: None for col in FEATURE_COLUMNS}

    if "entry" not in patient_data:
        return vitals

    for entry in patient_data["entry"]:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Observation":
            value_quantity = resource.get("valueQuantity", {})
            coding_list = resource.get("code", {}).get("coding", [])
            for coding in coding_list:
                loinc_code = coding.get("code")
                if loinc_code in LOINC_CODES:
                    vital_name = LOINC_CODES[loinc_code]
                    if isinstance(value_quantity, dict) and "value" in value_quantity:
                        vitals[vital_name] = float(value_quantity["value"])

            # Blood pressure and other panels
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

    # sirs_score and qsofa_score are not LOINC, but may be present as Observation.valueInteger
    for entry in patient_data["entry"]:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Observation":
            code_text = resource.get("code", {}).get("text", "").lower()
            if "sirs" in code_text and "score" in code_text:
                val = resource.get("valueInteger")
                if val is not None:
                    vitals["sirs_score"] = float(val)
            if "qsofa" in code_text and "score" in code_text:
                val = resource.get("valueInteger")
                if val is not None:
                    vitals["qsofa_score"] = float(val)

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
    Flatten FHIR bundles into structured DataFrame with all features.
    """
    records = []
    for bundle in bundles:
        vitals = extract_vital_signs(bundle)
        sepsis_label = extract_sepsis_label(bundle)
        record = {col: vitals.get(col) for col in FEATURE_COLUMNS}
        record[TARGET_COLUMN] = sepsis_label
        records.append(record)
    df = pd.DataFrame.from_records(records, columns=[*FEATURE_COLUMNS, TARGET_COLUMN])
    return df


def apply_imputation_and_scaling(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply robust imputation and StandardScaler to dataset for all features.
    """
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    if df.empty:
        return df.copy(), StandardScaler()

    df_imputed = df.copy()
    for col in FEATURE_COLUMNS:
        series = pd.to_numeric(df_imputed[col], errors="coerce")
        fill_value = float(series.median()) if series.notna().any() else 0.0
        df_imputed[col] = series.fillna(fill_value)

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

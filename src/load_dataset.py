#!/usr/bin/env python3
"""
Unified data loader that intelligently selects between:
1. PhysioNet Challenge 2019 (preferred if available)
2. FHIR data via flatten_fhir.py (fallback)

This allows seamless switching between datasets without pipeline changes.
"""

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def load_dataset() -> Tuple[pd.DataFrame, str]:
    """
    Intelligently load dataset from available sources.

    Priority:
    1. PhysioNet Challenge 2019 (if CSV files present)
    2. FHIR data via flatten_fhir.py (fallback)

    Returns:
        (DataFrame, source_name) tuple
    """
    physionet_dir = Path("data/raw_physionet")
    fhir_dir = Path("data/raw_fhir/fhir")

    # Check for PhysioNet data first
    if physionet_dir.exists() and list(physionet_dir.glob("*.csv")):
        print("✓ PhysioNet Challenge 2019 data found. Loading...")
        try:
            from load_physionet_data import (
                apply_imputation_and_scaling,
                load_physionet_to_dataframe,
            )

            df, patient_labels = load_physionet_to_dataframe()
            df_processed, _ = apply_imputation_and_scaling(df)
            print(f"✓ Loaded {len(df_processed)} records from PhysioNet")
            return df_processed, "PhysioNet Challenge 2019"
        except Exception as e:
            print(f"✗ Failed to load PhysioNet: {e}")
            print("  Falling back to FHIR data...")

    # Fall back to FHIR data
    if fhir_dir.exists() and list(fhir_dir.glob("*.json")):
        print("✓ FHIR data found. Loading...")
        try:
            # Import load_fhir_bundles for FHIR loading
            from flatten_fhir import (
                apply_imputation_and_scaling,
                flatten_fhir_to_dataframe,
                load_fhir_bundles,
            )

            bundles = load_fhir_bundles()
            df, _ = flatten_fhir_to_dataframe(bundles)
            df_processed, _ = apply_imputation_and_scaling(df)
            print(f"✓ Loaded {len(df_processed)} records from FHIR")
            return df_processed, "FHIR (Synthea)"
        except Exception as e:
            print(f"✗ Failed to load FHIR: {e}")

    raise RuntimeError(
        "No dataset available!\n"
        "Please either:\n"
        "1. Download PhysioNet Challenge 2019 to data/raw_physionet/\n"
        "   https://physionet.org/content/challenge-2019/1.0.0/\n"
        "2. Generate FHIR data with Synthea:"
        " SYNTHEA_PATIENTS=5000 bash scripts/generate_synthea_data.sh"
    )


if __name__ == "__main__":
    df, source = load_dataset()
    print(f"\nDataset Statistics ({source}):")
    print(f"  Total records: {len(df)}")
    print(f"  Sepsis cases: {df['sepsis'].sum()}")
    print(f"  Healthy cases: {len(df) - df['sepsis'].sum()}")
    print(f"  Class balance: {df['sepsis'].mean():.2%} positive")

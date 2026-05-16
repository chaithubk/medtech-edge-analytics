"""Unit tests for the FHIR flattening pipeline."""

import json

import pandas as pd
import pytest

from src import flatten_fhir


def test_load_fhir_bundles_recurses_nested_directories(tmp_path, monkeypatch):
    """Bundle JSON files in nested folders should be discovered."""
    nested_dir = tmp_path / "fhir"
    nested_dir.mkdir(parents=True)

    bundle_path = nested_dir / "patient-1.json"
    bundle_path.write_text(
        json.dumps({"resourceType": "Bundle", "entry": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(flatten_fhir, "RAW_FHIR_DIR", tmp_path)

    bundles = flatten_fhir.load_fhir_bundles()

    assert len(bundles) == 1
    assert bundles[0]["resourceType"] == "Bundle"


def test_flatten_fhir_to_dataframe_empty_keeps_expected_schema():
    """Even with zero bundles, the output DataFrame should preserve expected columns."""
    df, _ = flatten_fhir.flatten_fhir_to_dataframe([])

    assert list(df.columns) == [*flatten_fhir.FEATURE_COLUMNS, flatten_fhir.TARGET_COLUMN]
    assert df.empty


def test_apply_imputation_and_scaling_empty_dataframe_is_noop():
    """Preprocessing should not fail on an empty, correctly-shaped DataFrame."""
    df = pd.DataFrame(columns=[*flatten_fhir.FEATURE_COLUMNS, flatten_fhir.TARGET_COLUMN])

    df_processed, scaler = flatten_fhir.apply_imputation_and_scaling(df)

    assert list(df_processed.columns) == [
        *flatten_fhir.FEATURE_COLUMNS,
        flatten_fhir.TARGET_COLUMN,
    ]
    assert df_processed.empty
    assert scaler is not None


def test_apply_imputation_and_scaling_raises_for_missing_feature_columns():
    """A clear validation error is raised when required feature columns are absent."""
    df = pd.DataFrame({"sepsis": [0, 1]})

    with pytest.raises(ValueError, match="Missing required feature columns"):
        flatten_fhir.apply_imputation_and_scaling(df)


def test_apply_imputation_and_scaling_empty_unshaped_dataframe_raises_clear_error():
    """An entirely unshaped DataFrame should raise validation error, not KeyError."""
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Missing required feature columns"):
        flatten_fhir.apply_imputation_and_scaling(df)


def test_apply_imputation_and_scaling_all_missing_features_is_handled():
    """All-NaN feature columns should be imputed safely and remain processable."""
    df = pd.DataFrame(
        {
            "hr": [None, None, None],
            "bp_sys": [None, None, None],
            "bp_dia": [None, None, None],
            "o2_sat": [None, None, None],
            "temperature": [None, None, None],
            "respiratory_rate": [None, None, None],
            "wbc": [None, None, None],
            "lactate": [None, None, None],
            "creatinine": [None, None, None],
            "sirs_score": [None, None, None],
            "qsofa_score": [None, None, None],
            "sepsis": [0, 1, 0],
        }
    )

    df_processed, scaler = flatten_fhir.apply_imputation_and_scaling(df)

    assert not df_processed[flatten_fhir.FEATURE_COLUMNS].isna().any().any()
    assert scaler is not None


def test_main_raises_clear_error_when_no_bundles_found(tmp_path, monkeypatch):
    """main should fail fast with a clear message when no bundles are available."""
    monkeypatch.setattr(flatten_fhir, "RAW_FHIR_DIR", tmp_path)
    monkeypatch.setattr(flatten_fhir, "PROCESSED_DIR", tmp_path / "processed")

    with pytest.raises(ValueError, match="No FHIR bundles found"):
        flatten_fhir.main()


def test_extract_vital_signs_reads_value_quantity_and_components():
    """Observation values should be read from valueQuantity and component fields."""
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": "8867-4"}]},
                    "valueQuantity": {"value": 88},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": "85354-9"}]},
                    "component": [
                        {
                            "code": {"coding": [{"code": "8480-6"}]},
                            "valueQuantity": {"value": 121},
                        }
                    ],
                }
            },
        ],
    }

    vitals = flatten_fhir.extract_vital_signs(bundle)

    assert vitals["hr"] == 88.0
    assert vitals["bp_sys"] == 121.0


def test_load_fhir_bundles_groups_non_bundle_resources(tmp_path, monkeypatch):
    """Standalone resources should be grouped into synthetic patient bundles."""
    resources = [
        {"resourceType": "Patient", "id": "p1"},
        {
            "resourceType": "Condition",
            "subject": {"reference": "Patient/p1"},
            "code": {"coding": [{"code": flatten_fhir.SEPSIS_SNOMED_CODE}]},
        },
        {
            "resourceType": "Observation",
            "subject": {"reference": "Patient/p1"},
            "code": {"coding": [{"code": "8867-4"}]},
            "valueQuantity": {"value": 90},
        },
    ]

    (tmp_path / "resources.json").write_text(json.dumps(resources), encoding="utf-8")
    monkeypatch.setattr(flatten_fhir, "RAW_FHIR_DIR", tmp_path)

    bundles = flatten_fhir.load_fhir_bundles()
    df, _ = flatten_fhir.flatten_fhir_to_dataframe(bundles)

    assert len(bundles) == 1
    assert len(df) == 1
    assert int(df.iloc[0][flatten_fhir.TARGET_COLUMN]) == 1
    assert float(df.iloc[0]["hr"]) == 90.0

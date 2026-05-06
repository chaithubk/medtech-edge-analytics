"""Contract schema validation tests.

Validates fixture payloads against the vendored JSON Schema at
``contracts/vitals/v2.0.json``.  Any mismatch between fixtures and the
pinned contract is a CI failure — this test acts as the drift guard.
"""

import json
import pathlib

import jsonschema
import pytest

# Paths are resolved relative to the repository root so this test works
# regardless of the working directory.
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SCHEMA_PATH = _REPO_ROOT / "contracts" / "vitals" / "v2.0.json"
_FIXTURES_PATH = _REPO_ROOT / "tests" / "fixtures" / "sample_vitals.json"


@pytest.fixture(scope="module")
def vitals_schema() -> dict:
    """Load the vendored v2.0 vitals contract schema."""
    return json.loads(_SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def sample_vitals_fixture() -> list:
    """Load the sample_vitals fixture file."""
    return json.loads(_FIXTURES_PATH.read_text())


class TestContractSchemaV2:
    """Validate that fixtures and inline payloads conform to the v2 contract."""

    def test_schema_file_exists(self) -> None:
        """The vendored schema file must be present."""
        assert _SCHEMA_PATH.exists(), f"Schema file not found: {_SCHEMA_PATH}"

    def test_contract_version_file_exists(self) -> None:
        """The contract version pin file must be present."""
        version_file = _REPO_ROOT / "contracts" / "VITALS_CONTRACT_VERSION.txt"
        assert version_file.exists(), f"Contract version file not found: {version_file}"
        assert version_file.read_text().strip() == "v2.0.0"

    def test_fixture_payloads_valid(self, vitals_schema: dict, sample_vitals_fixture: list) -> None:
        """Every payload in sample_vitals.json must validate against the schema."""
        validator = jsonschema.Draft7Validator(vitals_schema)
        for i, payload in enumerate(sample_vitals_fixture):
            errors = list(validator.iter_errors(payload))
            assert (
                not errors
            ), f"Payload[{i}] in sample_vitals.json fails schema validation:\n" + "\n".join(
                f"  - {e.json_path}: {e.message}" for e in errors
            )

    def test_healthy_payload_valid(self, vitals_schema: dict) -> None:
        """Inline healthy-patient payload must pass schema validation."""
        payload = {
            "version": "2.0",
            "patient_id": "patient-test-001",
            "scenario": "healthy",
            "scenario_stage": "healthy",
            "timestamp": 1712973600000,
            "hr": 72.0,
            "bp_sys": 118.0,
            "bp_dia": 76.0,
            "o2_sat": 98.0,
            "temperature": 36.8,
            "respiratory_rate": 14.0,
            "wbc": 7.0,
            "lactate": 0.8,
            "sirs_score": 0,
            "qsofa_score": 0,
            "sepsis_stage": "none",
            "sepsis_onset_ts": None,
            "quality": "good",
            "source": "synthea-simulator",
        }
        jsonschema.validate(payload, vitals_schema)

    def test_sepsis_payload_with_onset_ts_valid(self, vitals_schema: dict) -> None:
        """Payload with an integer sepsis_onset_ts must pass schema validation."""
        payload = {
            "version": "2.0",
            "patient_id": "patient-test-002",
            "scenario": "sepsis",
            "scenario_stage": "sepsis_onset",
            "timestamp": 1712973620000,
            "hr": 110.0,
            "bp_sys": 92.0,
            "bp_dia": 58.0,
            "o2_sat": 91.0,
            "temperature": 38.9,
            "respiratory_rate": 24.0,
            "wbc": 13.5,
            "lactate": 2.8,
            "sirs_score": 3,
            "qsofa_score": 2,
            "sepsis_stage": "sepsis",
            "sepsis_onset_ts": 1712973620000,
            "quality": "degraded",
            "source": "synthea-simulator",
        }
        jsonschema.validate(payload, vitals_schema)

    def test_missing_version_rejected(self, vitals_schema: dict) -> None:
        """Payload missing the version field must fail schema validation."""
        payload = {
            "patient_id": "patient-test-001",
            "scenario": "healthy",
            "scenario_stage": "healthy",
            "timestamp": 1712973600000,
            "hr": 72.0,
            "bp_sys": 118.0,
            "bp_dia": 76.0,
            "o2_sat": 98.0,
            "temperature": 36.8,
            "respiratory_rate": 14.0,
            "wbc": 7.0,
            "lactate": 0.8,
            "sirs_score": 0,
            "qsofa_score": 0,
            "sepsis_stage": "none",
            "sepsis_onset_ts": None,
            "quality": "good",
            "source": "synthea-simulator",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, vitals_schema)

    def test_wrong_version_rejected(self, vitals_schema: dict) -> None:
        """Payload with version != '2.0' must fail schema validation."""
        payload = {
            "version": "1.0",
            "patient_id": "patient-test-001",
            "scenario": "healthy",
            "scenario_stage": "healthy",
            "timestamp": 1712973600000,
            "hr": 72.0,
            "bp_sys": 118.0,
            "bp_dia": 76.0,
            "o2_sat": 98.0,
            "temperature": 36.8,
            "respiratory_rate": 14.0,
            "wbc": 7.0,
            "lactate": 0.8,
            "sirs_score": 0,
            "qsofa_score": 0,
            "sepsis_stage": "none",
            "sepsis_onset_ts": None,
            "quality": "good",
            "source": "synthea-simulator",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, vitals_schema)

    def test_invalid_scenario_stage_rejected(self, vitals_schema: dict) -> None:
        """Payload with an invalid scenario_stage enum value must fail."""
        payload = {
            "version": "2.0",
            "patient_id": "patient-test-001",
            "scenario": "healthy",
            "scenario_stage": "stable",  # not a valid enum value
            "timestamp": 1712973600000,
            "hr": 72.0,
            "bp_sys": 118.0,
            "bp_dia": 76.0,
            "o2_sat": 98.0,
            "temperature": 36.8,
            "respiratory_rate": 14.0,
            "wbc": 7.0,
            "lactate": 0.8,
            "sirs_score": 0,
            "qsofa_score": 0,
            "sepsis_stage": "none",
            "sepsis_onset_ts": None,
            "quality": "good",
            "source": "synthea-simulator",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, vitals_schema)

    def test_integer_quality_rejected(self, vitals_schema: dict) -> None:
        """quality must be a string; integer value must fail."""
        payload = {
            "version": "2.0",
            "patient_id": "patient-test-001",
            "scenario": "healthy",
            "scenario_stage": "healthy",
            "timestamp": 1712973600000,
            "hr": 72.0,
            "bp_sys": 118.0,
            "bp_dia": 76.0,
            "o2_sat": 98.0,
            "temperature": 36.8,
            "respiratory_rate": 14.0,
            "wbc": 7.0,
            "lactate": 0.8,
            "sirs_score": 0,
            "qsofa_score": 0,
            "sepsis_stage": "none",
            "sepsis_onset_ts": None,
            "quality": 95,  # must be a string
            "source": "synthea-simulator",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, vitals_schema)

    def test_additional_properties_rejected(self, vitals_schema: dict) -> None:
        """Extra fields not in the schema must be rejected (additionalProperties: false)."""
        payload = {
            "version": "2.0",
            "patient_id": "patient-test-001",
            "scenario": "healthy",
            "scenario_stage": "healthy",
            "timestamp": 1712973600000,
            "hr": 72.0,
            "bp_sys": 118.0,
            "bp_dia": 76.0,
            "o2_sat": 98.0,
            "temperature": 36.8,
            "respiratory_rate": 14.0,
            "wbc": 7.0,
            "lactate": 0.8,
            "sirs_score": 0,
            "qsofa_score": 0,
            "sepsis_stage": "none",
            "sepsis_onset_ts": None,
            "quality": "good",
            "source": "synthea-simulator",
            "extra_field": "should_fail",  # not in schema
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, vitals_schema)

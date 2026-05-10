"""Contract evolution tests.

Verifies that the analytics pipeline behaves correctly across contract
revisions — specifically that:

1. **Added optional fields** (MINOR upgrade): payloads with new unknown fields
   are accepted at runtime; unknown fields are stripped with a warning.
2. **Added required fields** (BREAKING upgrade): payloads missing a field that
   is required in our pinned contract are rejected with a clear error.
3. **Renamed / removed fields** (BREAKING upgrade): payloads using old field
   names that are absent from our required set are rejected.
4. **Enum / type changes** (BREAKING upgrade): payloads with values that fail
   enum or type constraints are rejected.
5. **MAJOR version bump** (BREAKING): payloads from a different MAJOR version
   are rejected by the runtime parser.
6. **Pipeline hard-fail** on missing/unreadable schema file.

These tests target the *runtime* behaviour of ``parse_vital`` and
``resolve_schema_path``, not the JSON Schema itself (those tests live in
``test_contract_schema_v2.py``).
"""

import json
import pathlib
from unittest.mock import patch

import jsonschema
import pytest

from src.mqtt import mqtt_payload
from src.utils.schema_loader import _ENV_VAR, resolve_schema_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_CANONICAL_SCHEMA = _REPO_ROOT / "contracts" / "vitals" / "vitals.schema.json"

_BASE_PAYLOAD = {
    "version": "2.1.1",
    "patient_id": "patient-evo-001",
    "scenario": "healthy",
    "scenario_stage": "healthy",
    "timestamp": 1712973600000,
    "hr": 80.0,
    "bp_sys": 120.0,
    "bp_dia": 80.0,
    "o2_sat": 97.0,
    "temperature": 37.0,
    "respiratory_rate": 16.0,
    "wbc": 7.5,
    "lactate": 0.9,
    "creatinine": 1.0,
    "altered_mentation": False,
    "sirs_score": 0,
    "qsofa_score": 0,
    "sepsis_stage": "none",
    "sepsis_onset_ts": None,
    "quality": "good",
    "source": "simulator",
}


def _payload(**overrides) -> str:
    """Return a JSON string of _BASE_PAYLOAD with given field overrides."""
    p = dict(_BASE_PAYLOAD)
    p.update(overrides)
    return json.dumps(p)


def _payload_without(*fields) -> str:
    """Return a JSON string of _BASE_PAYLOAD with given fields removed."""
    p = {k: v for k, v in _BASE_PAYLOAD.items() if k not in fields}
    return json.dumps(p)


# ---------------------------------------------------------------------------
# 1. Added optional fields (MINOR upgrade — forward compatibility)
# ---------------------------------------------------------------------------


class TestAddedOptionalFields:
    """MINOR contract revision adds new optional fields to the payload.

    The consumer must accept the payload (strip unknown fields, keep processing).
    """

    def test_single_unknown_field_accepted(self):
        """Payload with one extra field from a newer MINOR revision is accepted."""
        vital = mqtt_payload.parse_vital(_payload(new_optional_metric=42.0))
        assert vital["hr"] == 80.0
        assert "new_optional_metric" not in vital

    def test_multiple_unknown_fields_accepted(self):
        """Payload with several extra fields from a newer MINOR revision is accepted."""
        vital = mqtt_payload.parse_vital(_payload(extra_a="x", extra_b=1, extra_c=True))
        assert vital["hr"] == 80.0
        assert "extra_a" not in vital
        assert "extra_b" not in vital

    def test_all_required_fields_preserved_after_strip(self):
        """After stripping unknown fields all required fields remain intact."""
        vital = mqtt_payload.parse_vital(_payload(future_field="value"))
        for field in mqtt_payload._VITAL_REQUIRED_FIELDS:
            assert field in vital, f"Required field missing after strip: '{field}'"

    def test_minor_version_with_extra_fields_accepted(self):
        """Producer upgraded to MINOR contract (2.1.0) with extra fields; consumer accepts."""
        vital = mqtt_payload.parse_vital(_payload(version="2.1.0", new_vital_sign=99.5))
        assert vital["version"] == "2.1.0"
        assert "new_vital_sign" not in vital


# ---------------------------------------------------------------------------
# 2. Added required fields (BREAKING — missing field rejected)
# ---------------------------------------------------------------------------


class TestAddedRequiredFields:
    """Simulates a BREAKING contract change that adds a new required field.

    If our runtime required-field list is updated to include the new field,
    payloads without it must be rejected.
    """

    def test_missing_existing_required_field_rejected(self):
        """Payload missing an existing required field is rejected with clear error."""
        with pytest.raises(ValueError, match="Missing required field: 'hr'"):
            mqtt_payload.parse_vital(_payload_without("hr"))

    def test_missing_recently_required_field_rejected(self):
        """Missing any required contract field causes a clear rejection."""
        for field in ("wbc", "lactate", "respiratory_rate", "sirs_score", "qsofa_score"):
            with pytest.raises(ValueError, match=f"Missing required field: '{field}'"):
                mqtt_payload.parse_vital(_payload_without(field))

    def test_missing_version_rejected(self):
        """Payload with no version field is rejected (version is required)."""
        with pytest.raises(ValueError, match="Schema version mismatch"):
            mqtt_payload.parse_vital(_payload_without("version"))

    def test_missing_patient_id_rejected(self):
        with pytest.raises(ValueError, match="Missing required field: 'patient_id'"):
            mqtt_payload.parse_vital(_payload_without("patient_id"))


# ---------------------------------------------------------------------------
# 3. Renamed / removed fields (BREAKING — old field names rejected)
# ---------------------------------------------------------------------------


class TestRenamedOrRemovedFields:
    """BREAKING contract change renames or removes fields.

    The consumer must reject payloads that are missing the required (original)
    field, whether the old field has been renamed or removed.
    """

    def test_renamed_vital_field_original_absent_rejected(self):
        """If 'hr' were renamed to 'heart_rate', the old name being absent is rejected."""
        p = dict(_BASE_PAYLOAD)
        p["heart_rate"] = p.pop("hr")  # simulate rename: hr → heart_rate
        with pytest.raises(ValueError, match="Missing required field: 'hr'"):
            mqtt_payload.parse_vital(json.dumps(p))

    def test_renamed_score_field_original_absent_rejected(self):
        """If 'sirs_score' were renamed, missing it is rejected."""
        p = dict(_BASE_PAYLOAD)
        p["sirs_index"] = p.pop("sirs_score")
        with pytest.raises(ValueError, match="Missing required field: 'sirs_score'"):
            mqtt_payload.parse_vital(json.dumps(p))

    def test_removed_quality_field_rejected(self):
        """If 'quality' were removed from a payload it must be flagged as missing."""
        with pytest.raises(ValueError, match="Missing required field: 'quality'"):
            mqtt_payload.parse_vital(_payload_without("quality"))


# ---------------------------------------------------------------------------
# 4. Enum / type changes (BREAKING — wrong value rejected)
# ---------------------------------------------------------------------------


class TestEnumAndTypeChanges:
    """BREAKING contract change modifies enum values or field types."""

    def test_invalid_scenario_enum_rejected(self):
        """Payload with invalid scenario value must fail schema validation."""
        schema = json.loads(_CANONICAL_SCHEMA.read_text())
        payload = dict(_BASE_PAYLOAD)
        payload["scenario"] = "cardiac"  # not in enum
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, schema)

    def test_invalid_scenario_stage_enum_rejected(self):
        """Payload with invalid scenario_stage value must fail schema validation."""
        schema = json.loads(_CANONICAL_SCHEMA.read_text())
        payload = dict(_BASE_PAYLOAD)
        payload["scenario_stage"] = "stabilising"  # not in enum
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, schema)

    def test_string_hr_type_rejected_at_runtime(self):
        """Non-numeric 'hr' value is rejected during runtime range check."""
        with pytest.raises(ValueError, match="Non-numeric value for 'hr'"):
            mqtt_payload.parse_vital(_payload(hr="fast"))

    def test_integer_quality_type_rejected_by_schema(self):
        """Integer 'quality' (must be string) fails schema validation."""
        schema = json.loads(_CANONICAL_SCHEMA.read_text())
        payload = dict(_BASE_PAYLOAD)
        payload["quality"] = 3  # must be str
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, schema)

    def test_out_of_range_value_rejected(self):
        """Out-of-clinical-range value is rejected during runtime range check."""
        with pytest.raises(ValueError, match="out of range"):
            mqtt_payload.parse_vital(_payload(hr=500.0))


# ---------------------------------------------------------------------------
# 5. MAJOR version bump — BREAKING
# ---------------------------------------------------------------------------


class TestMajorVersionBreaking:
    """Payload version with a different MAJOR number is always BREAKING."""

    def test_major_3_rejected(self):
        """v3 payload is rejected at runtime."""
        with pytest.raises(ValueError, match="Schema version mismatch"):
            mqtt_payload.parse_vital(_payload(version="3.0.0"))

    def test_major_1_rejected(self):
        """v1 payload is rejected at runtime."""
        with pytest.raises(ValueError, match="Schema version mismatch"):
            mqtt_payload.parse_vital(_payload(version="1.9.9"))

    def test_major_0_rejected(self):
        """v0 payload (pre-release) is rejected at runtime."""
        with pytest.raises(ValueError, match="Schema version mismatch"):
            mqtt_payload.parse_vital(_payload(version="0.9.0"))


# ---------------------------------------------------------------------------
# 6. Pipeline hard-fail on missing schema
# ---------------------------------------------------------------------------


class TestPipelineSchemaHardFail:
    """The analytics pipeline must fail clearly when the contract schema is missing."""

    def test_missing_schema_raises_file_not_found(self, monkeypatch, tmp_path):
        """resolve_schema_path raises FileNotFoundError when schema is absent."""
        monkeypatch.setenv(_ENV_VAR, str(tmp_path / "nonexistent.json"))
        with pytest.raises(FileNotFoundError):
            resolve_schema_path()

    def test_unreadable_schema_raises_permission_error(self, monkeypatch, tmp_path):
        """resolve_schema_path raises PermissionError when schema is not readable."""
        schema_file = tmp_path / "vitals.json"
        schema_file.write_text("{}")
        monkeypatch.setenv(_ENV_VAR, str(schema_file))

        # Mock os.access to make this deterministic for both root and non-root
        # execution environments.
        with patch("src.utils.schema_loader.os.access", return_value=False):
            with pytest.raises(PermissionError):
                resolve_schema_path()

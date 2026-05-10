"""MQTT payload parsing and serialization for vital signs and predictions.

Telemetry Contract: v2.1.1 (pinned by tag/commit in vendored manifest).

Version handling (SemVer):
- Payloads must carry ``version`` in SemVer format: ``"MAJOR.MINOR.PATCH"``.
- MAJOR must equal 2 for this consumer build (different MAJOR = BREAKING, drop).
- Any MINOR or PATCH within MAJOR 2 is accepted (backward-compatible additions).
- Legacy two-part ``"2.0"`` is accepted with a deprecation warning (migration window).

Forward-compatibility:
- Unknown fields from newer contract revisions are stripped with a WARNING rather
  than causing a hard rejection, so the pipeline keeps processing when a producer
  upgrades to a MINOR contract release before the consumer is updated.
- All fields required by the pinned v2.1.1 contract must still be present.
"""

import json
from typing import Any, Dict

from src.utils.contract_compat import PINNED_CONTRACT_VERSION, is_compatible_version
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Pinned contract version for this build.  Payloads with the same MAJOR version
# (e.g. '2.2.0', '2.1.2') are accepted; different MAJOR is treated as BREAKING.
VITALS_SCHEMA_VERSION = PINNED_CONTRACT_VERSION  # '2.1.1'

# Required fields for v2 vital signs.
# sepsis_onset_ts is required by the contract but nullable (None = onset not yet determined).
_VITAL_REQUIRED_FIELDS = [
    "version",
    "patient_id",
    "scenario",
    "scenario_stage",
    "timestamp",
    "hr",
    "bp_sys",
    "bp_dia",
    "o2_sat",
    "temperature",
    "respiratory_rate",
    "wbc",
    "lactate",
    "creatinine",
    "altered_mentation",
    "sirs_score",
    "qsofa_score",
    "sepsis_stage",
    "sepsis_onset_ts",
    "quality",
    "source",
]

# All property names permitted by the v2 contract (additionalProperties: false).
_VITAL_ALLOWED_FIELDS: frozenset = frozenset(_VITAL_REQUIRED_FIELDS)

# Valid numeric ranges for vital signs (field: (min, max))
_VITAL_RANGES: Dict[str, tuple] = {
    "hr": (30.0, 180.0),
    "bp_sys": (60.0, 200.0),
    "bp_dia": (30.0, 130.0),
    "o2_sat": (50.0, 100.0),
    "temperature": (32.0, 42.0),
    "respiratory_rate": (5.0, 60.0),
    "wbc": (0.5, 100.0),
    "lactate": (0.1, 30.0),
    "creatinine": (0.1, 30.0),
    "sirs_score": (0.0, 4.0),
    "qsofa_score": (0.0, 3.0),
}

_PREDICTION_REQUIRED_FIELDS = [
    "risk_score",
    "risk_level",
    "confidence",
    "timestamp_ms",
    "features_used",
    "model_latency_ms",
]


def parse_vital(payload_str: str) -> dict:
    """Parse and validate a v2 JSON vital signs payload string.

        Enforces the pinned v2 contract with SemVer compatibility rules:
        - ``version`` must be compatible with pinned ``2.1.1`` (same MAJOR).
            Different MAJOR values are rejected.
        - Unknown fields are stripped with a warning (forward compatibility for
            MINOR/PATCH producer upgrades).
    - All required fields must be present.
    - ``sepsis_onset_ts`` must be ``None`` or an integer epoch-ms value.
    - Numeric vitals must fall within the expected clinical ranges.

    Args:
        payload_str: JSON-encoded string with v2 vital sign data.

    Returns:
        Validated vital signs dict.

    Raises:
        ValueError: If JSON is invalid, version is incompatible, required
            fields are missing, ``sepsis_onset_ts`` has an invalid type, or
            numeric values are out of the expected clinical range.
    """
    try:
        data: Dict[str, Any] = json.loads(payload_str)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse vital payload: %s", exc)
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    # --- Version contract enforcement (SemVer-aware) ---
    received_version: object = data.get("version")
    if not is_compatible_version(received_version):
        logger.error(
            "Vitals contract version incompatible: pinned='%s', received='%s'. "
            "Dropping message. Ensure producer and consumer are on compatible contract versions.",
            VITALS_SCHEMA_VERSION,
            received_version,
        )
        raise ValueError(
            f"Schema version mismatch: expected compatible version with '{VITALS_SCHEMA_VERSION}', "
            f"got '{received_version}'"
        )

    # --- Additional properties: warn and strip (forward-compatibility) ---
    # Unknown fields from newer MINOR contract revisions are silently stripped so
    # the pipeline keeps running when a producer is upgraded before this consumer.
    unknown_keys = set(data.keys()) - _VITAL_ALLOWED_FIELDS
    if unknown_keys:
        logger.warning(
            "Payload contains %d unknown field(s) not in pinned contract v%s: %s. "
            "Fields stripped. Producer may be on a newer MINOR contract revision.",
            len(unknown_keys),
            VITALS_SCHEMA_VERSION,
            sorted(unknown_keys),
        )
        for key in unknown_keys:
            del data[key]

    # --- Required field presence ---
    for field in _VITAL_REQUIRED_FIELDS:
        if field not in data:
            logger.warning("Missing required vital field: '%s'", field)
            raise ValueError(f"Missing required field: '{field}'")

    # --- sepsis_onset_ts: must be null or an integer epoch-ms value ---
    onset_ts = data.get("sepsis_onset_ts")
    if onset_ts is not None and not isinstance(onset_ts, int):
        logger.warning("Invalid sepsis_onset_ts value: %r", onset_ts)
        raise ValueError(
            f"'sepsis_onset_ts' must be null or an integer epoch-ms value, got: {onset_ts!r}"
        )

    # altered_mentation must be boolean in v2.1+ contract.
    if not isinstance(data.get("altered_mentation"), bool):
        logger.warning("Invalid altered_mentation value: %r", data.get("altered_mentation"))
        raise ValueError("'altered_mentation' must be a boolean value")

    # --- Numeric range validation ---
    for field, (lo, hi) in _VITAL_RANGES.items():
        try:
            value = float(data[field])
        except (TypeError, ValueError) as exc:
            logger.warning("Non-numeric value for vital field '%s': %s", field, data[field])
            raise ValueError(f"Non-numeric value for '{field}': {data[field]}") from exc
        if not (lo <= value <= hi):
            logger.warning("Vital field '%s' out of range [%s, %s]: %s", field, lo, hi, value)
            raise ValueError(f"Value for '{field}' out of range [{lo}, {hi}]: {value}")

    return data


def serialize_prediction(prediction: dict) -> str:
    """Serialize a prediction dict to a pretty-printed JSON string.

    Args:
        prediction: Dict with keys: risk_score, risk_level, confidence,
            timestamp_ms, features_used, model_latency_ms.

    Returns:
        JSON string with 2-space indentation.

    Raises:
        ValueError: If required fields are missing from prediction.
    """
    for field in _PREDICTION_REQUIRED_FIELDS:
        if field not in prediction:
            raise ValueError(f"Missing required prediction field: '{field}'")
    return json.dumps(prediction, indent=2)

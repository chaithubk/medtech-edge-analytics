"""MQTT payload parsing and serialization for vital signs and predictions.

Telemetry Contract: v2.0
This module enforces the v2 MQTT payload schema published by medtech-vitals-publisher.
Payloads must carry an explicit ``version`` field equal to ``"2.0"`` and must not
contain fields outside the contract's property set (``additionalProperties: false``).
"""

import json
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Enforced schema version — messages with any other value are dropped.
VITALS_SCHEMA_VERSION = "2.0"

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

    Enforces the strict v2 contract:
    - ``version`` must equal ``"2.0"`` exactly (string); missing or any other
      value causes immediate rejection.
    - No fields outside the contract's property set are permitted.
    - All required fields must be present.
    - ``sepsis_onset_ts`` must be ``None`` or an integer epoch-ms value.
    - Numeric vitals must fall within the expected clinical ranges.

    Args:
        payload_str: JSON-encoded string with v2 vital sign data.

    Returns:
        Validated vital signs dict.

    Raises:
        ValueError: If JSON is invalid, version is wrong, unknown fields are
            present, required fields are missing, ``sepsis_onset_ts`` has an
            invalid type, or numeric values are out of the expected clinical range.
    """
    try:
        data: Dict[str, Any] = json.loads(payload_str)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse vital payload: %s", exc)
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    # --- Strict version contract enforcement ---
    received_version = data.get("version")
    if received_version != VITALS_SCHEMA_VERSION:
        logger.error(
            "Vitals schema version mismatch: expected '%s', received '%s'. "
            "Dropping message. Update the publisher to emit v2 payloads.",
            VITALS_SCHEMA_VERSION,
            received_version,
        )
        raise ValueError(
            f"Schema version mismatch: expected '{VITALS_SCHEMA_VERSION}', "
            f"got '{received_version}'"
        )

    # --- Additional properties check (contract sets additionalProperties: false) ---
    unknown_keys = set(data.keys()) - _VITAL_ALLOWED_FIELDS
    if unknown_keys:
        logger.warning("Unknown fields in vital payload: %s", sorted(unknown_keys))
        raise ValueError(f"Unknown fields not permitted by contract: {sorted(unknown_keys)}")

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

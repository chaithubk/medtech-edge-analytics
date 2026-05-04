"""MQTT payload parsing and serialization for vital signs and predictions.

Telemetry Contract: v2.0
This module enforces the v2 MQTT payload schema published by medtech-vitals-publisher.
Messages with version != "2.0" are rejected (logged and dropped) without crashing.
"""

import json
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Enforced schema version — messages with any other value are dropped.
VITALS_SCHEMA_VERSION = "2.0"

# Required fields for v2 vital signs (non-nullable)
_VITAL_REQUIRED_FIELDS = [
    "version",
    "patient_id",
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
    "quality",
    "source",
]

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

    Enforces strict contract: ``version`` must equal ``"2.0"``.  Messages with a
    missing or mismatched version are logged as errors and a ``ValueError`` is
    raised so the caller can drop the message without crashing.

    Args:
        payload_str: JSON-encoded string with v2 vital sign data.

    Returns:
        Validated vital signs dict.  ``sepsis_onset_ts`` may be ``None``.

    Raises:
        ValueError: If JSON is invalid, version is wrong, fields are missing,
            or numeric values are out of the expected clinical range.
    """
    try:
        data: Dict[str, Any] = json.loads(payload_str)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse vital payload: %s", exc)
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    # --- Version contract enforcement ---
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

    # --- Required field presence ---
    for field in _VITAL_REQUIRED_FIELDS:
        if field not in data:
            logger.warning("Missing required vital field: '%s'", field)
            raise ValueError(f"Missing required field: '{field}'")

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

    # sepsis_onset_ts is allowed to be null/absent (onset not yet determined)
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

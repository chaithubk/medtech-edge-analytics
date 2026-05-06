"""MQTT payload parsing and serialization for vital signs and predictions.

Telemetry Contract: v2.0
This module validates the v2 MQTT payload schema published by medtech-vitals-publisher.
For compatibility with publisher variants, schema version may be supplied under
supported alias keys or inferred from a complete v2 field set.
"""

import json
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Enforced schema version — messages with any other value are dropped.
VITALS_SCHEMA_VERSION = "2.0"

# Accepted keys that may carry schema version in different publisher builds.
_VERSION_KEYS = ("version", "schema_version", "payload_version", "contract_version")

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

    Validates v2 contract compatibility. The version can be provided as
    ``version`` or a supported alias key. If version is omitted but all required
    v2 fields are present, version is inferred as ``"2.0"`` and processing
    continues.

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

    # --- Version contract enforcement with compatibility aliases ---
    received_version = None
    for key in _VERSION_KEYS:
        value = data.get(key)
        if value is not None:
            received_version = str(value)
            break

    allowed_versions = {"2", "2.0"}
    if received_version not in allowed_versions:
        missing_version = received_version is None
        # Some publisher versions omit explicit version but still emit full v2 fields.
        if missing_version and all(
            field in data for field in _VITAL_REQUIRED_FIELDS if field != "version"
        ):
            logger.warning(
                "Vitals schema version missing; inferred '%s' from v2 field set",
                VITALS_SCHEMA_VERSION,
            )
            data["version"] = VITALS_SCHEMA_VERSION
        else:
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
    else:
        # Canonicalize the field for downstream traceability and logging.
        data["version"] = VITALS_SCHEMA_VERSION

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

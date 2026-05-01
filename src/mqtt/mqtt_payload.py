"""MQTT payload parsing and serialization for vital signs and predictions."""

import json
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Required fields for vital signs
_VITAL_REQUIRED_FIELDS = ["timestamp", "hr", "bp_sys", "bp_dia", "o2_sat", "temperature", "quality"]

# Valid numeric ranges for vital signs
_VITAL_RANGES: Dict[str, tuple] = {
    "hr": (30.0, 180.0),
    "bp_sys": (60.0, 200.0),
    "bp_dia": (30.0, 130.0),
    "o2_sat": (50.0, 100.0),
    "temperature": (32.0, 42.0),
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
    """Parse a JSON vital signs payload string.

    Args:
        payload_str: JSON-encoded string with vital sign data.

    Returns:
        Validated vital signs dict.

    Raises:
        ValueError: If JSON is invalid, fields are missing, or values are out of range.
    """
    try:
        data: Dict[str, Any] = json.loads(payload_str)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse vital payload: %s", exc)
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    for field in _VITAL_REQUIRED_FIELDS:
        if field not in data:
            logger.warning("Missing required vital field: '%s'", field)
            raise ValueError(f"Missing required field: '{field}'")

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

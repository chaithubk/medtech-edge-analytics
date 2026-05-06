"""Sepsis risk scoring using TFLite model and engineered features."""

import time
from typing import TYPE_CHECKING

import numpy as np

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.inference.tflite_model import TFLiteModel
    from src.inference.vital_buffer import VitalBuffer

logger = get_logger(__name__)

# Small epsilon to avoid division by zero during feature normalization
_EPSILON = 1e-7

# Pre-computed normalization statistics (from training data)
# Shape: (20,) - one entry per feature in the order returned by VitalBuffer.get_all_features()
#
# Feature positions:
#  0- 4: hr_mean, hr_std, hr_min, hr_max, hr_trend
#  5- 9: bp_sys_mean, bp_sys_std, bp_sys_min, bp_sys_max, bp_sys_trend
# 10-14: bp_dia_mean, bp_dia_std, bp_dia_min, bp_dia_max, bp_dia_trend
#    15: o2_mean
#    16: rr_mean           (respiratory rate — v2)
#    17: rr_trend          (respiratory rate dynamics — v2)
#    18: lactate_mean      (mmol/L — v2)
#    19: sirs_qsofa_mean   (SIRS + qSOFA composite 0-7 — v2)
_NORM_MEANS = np.array(
    [
        85.0,
        10.0,
        60.0,
        110.0,
        0.0,  # hr: mean, std, min, max, trend
        120.0,
        15.0,
        90.0,
        150.0,
        0.0,  # bp_sys
        80.0,
        8.0,
        60.0,
        100.0,
        0.0,  # bp_dia
        96.0,  # o2_mean
        16.0,  # rr_mean  (normal ~12-20 bpm)
        0.0,  # rr_trend
        1.2,  # lactate_mean  (normal <2 mmol/L)
        1.5,  # sirs_qsofa_mean  (combined, normal ~0-1)
    ],
    dtype=np.float32,
)
_NORM_STDS = np.array(
    [
        15.0,
        5.0,
        15.0,
        15.0,
        1.0,  # hr
        20.0,
        8.0,
        15.0,
        20.0,
        1.0,  # bp_sys
        10.0,
        4.0,
        10.0,
        10.0,
        1.0,  # bp_dia
        2.0,  # o2_mean
        5.0,  # rr_mean
        1.0,  # rr_trend
        1.5,  # lactate_mean
        2.0,  # sirs_qsofa_mean
    ],
    dtype=np.float32,
)


def _classify_risk(risk_score: float) -> str:
    """Classify risk level from percentage score.

    Args:
        risk_score: Float 0-100.

    Returns:
        'LOW', 'MODERATE', or 'HIGH'.
    """
    if risk_score < 30.0:
        return "LOW"
    if risk_score <= 70.0:
        return "MODERATE"
    return "HIGH"


class SepsisScorer:
    """Sepsis risk scoring using engineered features and TFLite inference."""

    def __init__(self, model: "TFLiteModel") -> None:
        """Initialize scorer with a loaded TFLiteModel.

        Args:
            model: A TFLiteModel instance (may or may not be loaded).
        """
        self._model = model

    def score(self, vital_buffer: "VitalBuffer") -> dict:
        """Run the full inference pipeline and return a risk assessment.

        Args:
            vital_buffer: VitalBuffer with recent vital history.

        Returns:
            Dict with keys: risk_score, risk_level, confidence, timestamp_ms,
            features_used, model_latency_ms.
        """
        if not vital_buffer.get_history():
            logger.warning("Vital buffer is empty, returning default risk")
            return self._default_result()

        if not vital_buffer.is_full():
            logger.warning(
                "Buffer not full (%d/%d), scoring with available data",
                len(vital_buffer.get_history()),
                vital_buffer.get_size(),
            )

        try:
            features = vital_buffer.get_all_features()  # (1, 20) float32
            normalized = self._normalize(features)

            start = time.time()
            raw_score = self._model.infer(normalized)
            latency_ms = (time.time() - start) * 1000

            risk_score = float(raw_score) * 100.0
            risk_level = _classify_risk(risk_score)

            return {
                "risk_score": round(risk_score, 2),
                "risk_level": risk_level,
                "confidence": round(float(raw_score), 4),
                "timestamp_ms": int(time.time() * 1000),
                "features_used": 20,
                "model_latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            logger.error("Scoring failed: %s", exc)
            return self._default_result()

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        """Normalize features using pre-computed statistics.

        Args:
            features: np.ndarray of shape (1, 20).

        Returns:
            Normalized np.ndarray of shape (1, 20).
        """
        return (features - _NORM_MEANS) / (_NORM_STDS + _EPSILON)

    @staticmethod
    def _default_result() -> dict:
        """Return a default moderate-risk result when scoring fails."""
        return {
            "risk_score": 50.0,
            "risk_level": "MODERATE",
            "confidence": 0.5,
            "timestamp_ms": int(time.time() * 1000),
            "features_used": 20,
            "model_latency_ms": 0.0,
        }

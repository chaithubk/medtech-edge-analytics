"""Vital history circular buffer for sepsis detection."""

from collections import deque

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VitalBuffer:
    """Circular buffer for vital history (360 points = 1 hour @ 10s)."""

    REQUIRED_FIELDS = ["hr", "bp_sys", "bp_dia", "o2_sat", "temperature"]

    def __init__(self, size: int = 360) -> None:
        """Initialize circular buffer.

        Args:
            size: Maximum number of vital readings to store.
        """
        self._size = size
        self._buffer: deque = deque(maxlen=size)

    def add_vital(self, vital: dict) -> None:
        """Add a vital reading to the buffer.

        Args:
            vital: Dict with keys: hr, bp_sys, bp_dia, o2_sat, temperature.
        """
        for field in self.REQUIRED_FIELDS:
            if field not in vital:
                logger.warning("Missing field '%s' in vital, skipping", field)
                return
            try:
                float(vital[field])
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid numeric value for '%s': %s, skipping", field, vital[field]
                )
                return
        self._buffer.append(vital)

    def is_full(self) -> bool:
        """Return True if buffer contains the maximum number of points."""
        return len(self._buffer) >= self._size

    def get_stats(self, window_size: int = 60) -> dict:
        """Compute rolling statistics over last window_size points.

        Args:
            window_size: Number of most recent points to include.

        Returns:
            Dict with mean, std, min, max, trend for each vital sign.
        """
        history = list(self._buffer)
        window = history[-window_size:] if len(history) >= window_size else history
        if not window:
            return {}

        def _arr(key: str) -> np.ndarray:
            return np.array([v[key] for v in window], dtype=np.float64)

        def _trend(arr: np.ndarray) -> float:
            if len(arr) < 2:
                return 0.0
            coeffs = np.polyfit(np.arange(len(arr)), arr, 1)
            return float(coeffs[0])

        hr = _arr("hr")
        bp_sys = _arr("bp_sys")
        bp_dia = _arr("bp_dia")
        o2 = _arr("o2_sat")
        temp = _arr("temperature")

        return {
            "hr_mean": float(np.mean(hr)),
            "hr_std": float(np.std(hr)),
            "hr_min": float(np.min(hr)),
            "hr_max": float(np.max(hr)),
            "hr_trend": _trend(hr),
            "bp_sys_mean": float(np.mean(bp_sys)),
            "bp_sys_std": float(np.std(bp_sys)),
            "bp_sys_min": float(np.min(bp_sys)),
            "bp_sys_max": float(np.max(bp_sys)),
            "bp_sys_trend": _trend(bp_sys),
            "bp_dia_mean": float(np.mean(bp_dia)),
            "bp_dia_std": float(np.std(bp_dia)),
            "bp_dia_min": float(np.min(bp_dia)),
            "bp_dia_max": float(np.max(bp_dia)),
            "bp_dia_trend": _trend(bp_dia),
            "o2_mean": float(np.mean(o2)),
            "o2_std": float(np.std(o2)),
            "o2_min": float(np.min(o2)),
            "o2_max": float(np.max(o2)),
            "o2_trend": _trend(o2),
            "temp_mean": float(np.mean(temp)),
            "temp_trend": _trend(temp),
        }

    def get_history(self) -> list:
        """Return all vitals currently in the buffer."""
        return list(self._buffer)

    def get_all_features(self) -> np.ndarray:
        """Return 20-feature vector shaped (1, 20) for TFLite model input.

        Feature order (CRITICAL - must match model):
          0-4:   hr_mean, hr_std, hr_min, hr_max, hr_trend
          5-9:   bp_sys_mean, bp_sys_std, bp_sys_min, bp_sys_max, bp_sys_trend
          10-14: bp_dia_mean, bp_dia_std, bp_dia_min, bp_dia_max, bp_dia_trend
          15-19: o2_mean, o2_std, o2_min, o2_max, o2_trend

        Returns:
            np.ndarray of shape (1, 20) dtype float32.
        """
        stats = self.get_stats()
        if not stats:
            logger.warning("Empty buffer, returning zero features")
            return np.zeros((1, 20), dtype=np.float32)

        features = np.array(
            [
                stats["hr_mean"], stats["hr_std"], stats["hr_min"], stats["hr_max"], stats["hr_trend"],
                stats["bp_sys_mean"], stats["bp_sys_std"], stats["bp_sys_min"], stats["bp_sys_max"], stats["bp_sys_trend"],
                stats["bp_dia_mean"], stats["bp_dia_std"], stats["bp_dia_min"], stats["bp_dia_max"], stats["bp_dia_trend"],
                stats["o2_mean"], stats["o2_std"], stats["o2_min"], stats["o2_max"], stats["o2_trend"],
            ],
            dtype=np.float32,
        ).reshape(1, 20)
        return features

"""Tests for vital history circular buffer."""

import numpy as np
import pytest

from src.inference.vital_buffer import VitalBuffer


def _make_vital(hr=80.0, bp_sys=120.0, bp_dia=80.0, o2_sat=97.0, temperature=37.0, ts_offset=0):
    return {
        "timestamp": 1712973600000 + ts_offset * 10000,
        "hr": hr,
        "bp_sys": bp_sys,
        "bp_dia": bp_dia,
        "o2_sat": o2_sat,
        "temperature": temperature,
        "quality": 95,
        "source": "simulator",
    }


class TestVitalBuffer:
    def test_add_vital_single(self):
        buf = VitalBuffer()
        buf.add_vital(_make_vital())
        assert len(buf.get_history()) == 1

    def test_add_vital_multiple(self):
        buf = VitalBuffer()
        for i in range(10):
            buf.add_vital(_make_vital(ts_offset=i))
        assert len(buf.get_history()) == 10

    def test_circular_wrap(self):
        buf = VitalBuffer(size=360)
        for i in range(361):
            buf.add_vital(_make_vital(hr=float(i), ts_offset=i))
        assert len(buf.get_history()) == 360
        hrs = [v["hr"] for v in buf.get_history()]
        assert 0.0 not in hrs
        assert 360.0 in hrs

    def test_is_full_false_when_partial(self):
        buf = VitalBuffer(size=360)
        for i in range(10):
            buf.add_vital(_make_vital(ts_offset=i))
        assert buf.is_full() is False

    def test_is_full_true_when_complete(self):
        buf = VitalBuffer(size=360)
        for i in range(360):
            buf.add_vital(_make_vital(ts_offset=i))
        assert buf.is_full() is True

    def test_get_stats_full(self):
        buf = VitalBuffer(size=360)
        for i in range(360):
            buf.add_vital(_make_vital(hr=80.0, bp_sys=120.0, bp_dia=80.0, o2_sat=97.0, temperature=37.0, ts_offset=i))
        stats = buf.get_stats()
        assert abs(stats["hr_mean"] - 80.0) < 0.001
        assert "hr_std" in stats
        assert "hr_trend" in stats

    def test_get_stats_partial(self):
        buf = VitalBuffer(size=360)
        for i in range(30):
            buf.add_vital(_make_vital(hr=75.0, ts_offset=i))
        stats = buf.get_stats()
        assert abs(stats["hr_mean"] - 75.0) < 0.001
        assert "o2_mean" in stats

    def test_get_features_shape(self):
        buf = VitalBuffer(size=360)
        for i in range(10):
            buf.add_vital(_make_vital(ts_offset=i))
        features = buf.get_all_features()
        assert features.shape == (1, 20)
        assert features.dtype == np.float32

    def test_missing_vital_field(self):
        buf = VitalBuffer()
        bad_vital = {"timestamp": 1000, "hr": 80.0}  # missing required fields
        buf.add_vital(bad_vital)
        assert len(buf.get_history()) == 0

    def test_get_features_empty_buffer(self):
        buf = VitalBuffer()
        features = buf.get_all_features()
        assert features.shape == (1, 20)
        assert np.all(features == 0.0)
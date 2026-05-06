"""Tests for vital history circular buffer."""

import numpy as np

from src.inference.vital_buffer import VitalBuffer


def _make_vital(
    hr=80.0,
    bp_sys=120.0,
    bp_dia=80.0,
    o2_sat=97.0,
    temperature=37.0,
    respiratory_rate=16.0,
    wbc=7.5,
    lactate=0.9,
    sirs_score=0,
    qsofa_score=0,
    ts_offset=0,
):
    return {
        "version": "2.0",
        "patient_id": "patient-test-001",
        "scenario": "healthy",
        "scenario_stage": "healthy",
        "timestamp": 1712973600000 + ts_offset * 10000,
        "hr": hr,
        "bp_sys": bp_sys,
        "bp_dia": bp_dia,
        "o2_sat": o2_sat,
        "temperature": temperature,
        "respiratory_rate": respiratory_rate,
        "wbc": wbc,
        "lactate": lactate,
        "sirs_score": sirs_score,
        "qsofa_score": qsofa_score,
        "sepsis_stage": "none",
        "sepsis_onset_ts": None,
        "quality": "good",
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
            buf.add_vital(
                _make_vital(
                    hr=80.0,
                    bp_sys=120.0,
                    bp_dia=80.0,
                    o2_sat=97.0,
                    temperature=37.0,
                    respiratory_rate=16.0,
                    wbc=7.5,
                    lactate=0.9,
                    ts_offset=i,
                )
            )
        stats = buf.get_stats()
        assert abs(stats["hr_mean"] - 80.0) < 0.001
        assert "hr_std" in stats
        assert "hr_trend" in stats
        assert "rr_mean" in stats
        assert "lactate_mean" in stats
        assert "sirs_score_mean" in stats
        assert "qsofa_score_mean" in stats

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

    def test_get_features_clinical_values(self):
        """Feature positions 15-19 should reflect v2 clinical fields."""
        buf = VitalBuffer(size=360)
        for i in range(10):
            buf.add_vital(
                _make_vital(
                    o2_sat=96.0,
                    respiratory_rate=20.0,
                    lactate=2.0,
                    sirs_score=2,
                    qsofa_score=1,
                    ts_offset=i,
                )
            )
        features = buf.get_all_features()
        assert abs(float(features[0, 15]) - 96.0) < 0.01  # o2_mean
        assert abs(float(features[0, 16]) - 20.0) < 0.01  # rr_mean
        # rr_trend (17) near 0 since RR is constant
        assert abs(float(features[0, 17])) < 0.1
        assert abs(float(features[0, 18]) - 2.0) < 0.01  # lactate_mean
        # sirs_qsofa composite = sirs_score_mean (2.0) + qsofa_score_mean (1.0) = 3.0
        assert abs(float(features[0, 19]) - 3.0) < 0.01

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

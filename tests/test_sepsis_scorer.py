"""Tests for sepsis risk scorer."""

import time
from unittest.mock import MagicMock

import pytest

from src.inference.sepsis_scorer import SepsisScorer, _classify_risk
from src.inference.vital_buffer import VitalBuffer


def _make_vital(
    hr,
    bp_sys,
    bp_dia,
    o2_sat,
    temperature,
    idx=0,
    respiratory_rate=16.0,
    wbc=7.5,
    lactate=0.9,
    sirs_score=0.0,
    qsofa_score=0.0,
):
    return {
        "version": "2.0",
        "patient_id": "patient-test-001",
        "scenario": "healthy",
        "scenario_stage": "healthy",
        "timestamp": 1712973600000 + idx * 10000,
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


def _fill_buffer(buf, vital, n=60):
    for i in range(n):
        buf.add_vital({**vital, "timestamp": 1712973600000 + i * 10000})


def _mock_model(return_value: float) -> MagicMock:
    m = MagicMock()
    m.infer.return_value = return_value
    return m


class TestSepsisScorer:
    def test_score_returns_dict(self):
        buf = VitalBuffer()
        _fill_buffer(buf, _make_vital(80, 120, 80, 97, 37.0))
        scorer = SepsisScorer(_mock_model(0.1))
        result = scorer.score(buf)
        assert isinstance(result, dict)
        assert "risk_score" in result
        assert "risk_level" in result
        assert "confidence" in result
        assert "timestamp_ms" in result
        assert "features_used" in result
        assert "model_latency_ms" in result

    def test_score_healthy(self):
        buf = VitalBuffer()
        _fill_buffer(buf, _make_vital(80, 120, 80, 97, 37.0))
        scorer = SepsisScorer(_mock_model(0.1))
        result = scorer.score(buf)
        assert result["risk_score"] < 30.0
        assert result["risk_level"] == "LOW"

    def test_score_sepsis(self):
        buf = VitalBuffer()
        _fill_buffer(
            buf,
            _make_vital(
                125,
                95,
                55,
                88,
                39.5,
                respiratory_rate=26.0,
                wbc=14.5,
                lactate=3.2,
                sirs_score=3.0,
                qsofa_score=2.0,
            ),
        )
        scorer = SepsisScorer(_mock_model(0.9))
        result = scorer.score(buf)
        assert result["risk_score"] > 70.0
        assert result["risk_level"] == "HIGH"

    def test_score_moderate(self):
        buf = VitalBuffer()
        _fill_buffer(buf, _make_vital(100, 110, 70, 93, 38.0))
        scorer = SepsisScorer(_mock_model(0.5))
        result = scorer.score(buf)
        assert 30.0 <= result["risk_score"] <= 70.0
        assert result["risk_level"] == "MODERATE"

    def test_risk_level_low(self):
        assert _classify_risk(0.0) == "LOW"
        assert _classify_risk(29.9) == "LOW"

    def test_risk_level_moderate(self):
        assert _classify_risk(30.0) == "MODERATE"
        assert _classify_risk(70.0) == "MODERATE"

    def test_risk_level_high(self):
        assert _classify_risk(70.1) == "HIGH"
        assert _classify_risk(100.0) == "HIGH"

    def test_confidence_range(self):
        buf = VitalBuffer()
        _fill_buffer(buf, _make_vital(80, 120, 80, 97, 37.0))
        scorer = SepsisScorer(_mock_model(0.65))
        result = scorer.score(buf)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_features_used(self):
        buf = VitalBuffer()
        _fill_buffer(buf, _make_vital(80, 120, 80, 97, 37.0))
        scorer = SepsisScorer(_mock_model(0.2))
        result = scorer.score(buf)
        assert result["features_used"] == 20

    @pytest.mark.slow
    def test_latency(self):
        buf = VitalBuffer()
        _fill_buffer(buf, _make_vital(80, 120, 80, 97, 37.0))
        scorer = SepsisScorer(_mock_model(0.2))
        # Warm-up call to avoid cold-start overhead
        scorer.score(buf)
        start = time.time()
        scorer.score(buf)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 100.0, (
            f"Scoring took {elapsed_ms:.1f}ms (spec requirement: <100ms). "
            "This may indicate a slow CI environment or a regression."
        )

    def test_score_empty_buffer(self):
        buf = VitalBuffer()
        scorer = SepsisScorer(_mock_model(0.5))
        result = scorer.score(buf)
        assert result["risk_score"] == 50.0
        assert result["risk_level"] == "MODERATE"

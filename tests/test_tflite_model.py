"""Tests for TFLite model wrapper."""

import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.inference.tflite_model import TFLiteModel

MODEL_PATH = "models/sepsis_model.tflite"


class TestTFLiteModel:
    def test_load_model_success(self):
        model = TFLiteModel(MODEL_PATH)
        result = model.load()
        assert result is True
        assert model.is_loaded() is True

    def test_load_model_missing(self):
        model = TFLiteModel("/nonexistent/path/model.tflite")
        result = model.load()
        assert result is False
        assert model.is_loaded() is False

    def test_inference_valid_input(self):
        model = TFLiteModel(MODEL_PATH)
        model.load()
        features = np.zeros((1, 20), dtype=np.float32)
        result = model.infer(features)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_inference_invalid_shape(self):
        model = TFLiteModel(MODEL_PATH)
        model.load()
        bad_input = np.zeros((1, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            model.infer(bad_input)

    def test_latency(self):
        model = TFLiteModel(MODEL_PATH)
        model.load()
        features = np.zeros((1, 20), dtype=np.float32)
        # Warm-up call to avoid cold-start overhead
        model.infer(features)
        start = time.time()
        model.infer(features)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 100.0, (
            f"Inference took {elapsed_ms:.1f}ms (spec requirement: <100ms). "
            "This may indicate a slow CI environment or a regression."
        )

    def test_input_output_shapes(self):
        model = TFLiteModel(MODEL_PATH)
        model.load()
        assert model.get_input_shape() == (1, 20)
        assert model.get_output_shape() == (1, 1)

    def test_inference_without_loading(self):
        model = TFLiteModel(MODEL_PATH)
        # Do not call load()
        features = np.zeros((1, 20), dtype=np.float32)
        result = model.infer(features)
        assert result == 0.5
"""TensorFlow Lite model inference wrapper."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TFLiteModel:
    """TensorFlow Lite model inference wrapper."""

    from typing import Any, Dict, List

    def __init__(self, model_path: str) -> None:
        """Initialize with path to .tflite model file.

        Args:
            model_path: File path to the .tflite model.
        """
        self._model_path = model_path
        self._interpreter: Optional[Any] = None
        self._input_details: Optional[List[Dict[str, Any]]] = None
        self._output_details: Optional[List[Dict[str, Any]]] = None

    def load(self) -> bool:
        """Load TFLite model from file and allocate tensors.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            import tensorflow as tf

            self._interpreter = tf.lite.Interpreter(model_path=self._model_path)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            logger.info("TFLite model loaded from %s", self._model_path)
            return True
        except FileNotFoundError:
            logger.error("Model file not found: %s", self._model_path)
            return False
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return False

    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        return self._interpreter is not None

    def infer(self, features: np.ndarray) -> float:
        """Run inference on feature vector.

        Args:
            features: np.ndarray of shape (1, 20) float32.

        Returns:
            Sepsis probability in range [0.0, 1.0].

        Raises:
            ValueError: If input shape does not match expected (1, 20).
        """
        if not self.is_loaded():
            logger.error("Model not loaded, returning neutral prediction")
            return 0.5

        # Type-safe asserts for mypy
        assert self._interpreter is not None, "Interpreter is not loaded"
        assert self._input_details is not None, "Input details not loaded"
        assert self._output_details is not None, "Output details not loaded"

        expected_shape = tuple(self._input_details[0]["shape"])
        if features.shape != expected_shape:
            raise ValueError(
                f"Input shape mismatch: expected {expected_shape}, got {features.shape}"
            )

        try:
            start = time.time()
            self._interpreter.set_tensor(
                self._input_details[0]["index"],
                features.astype(np.float32),
            )
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_details[0]["index"])
            latency_ms = (time.time() - start) * 1000
            logger.debug("Inference latency: %.1f ms", latency_ms)
            return float(output.flatten()[0])
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return 0.5

    def get_input_shape(self) -> tuple:
        """Return expected input shape."""
        if self._input_details:
            return tuple(self._input_details[0]["shape"])
        return (1, 20)

    def get_output_shape(self) -> tuple:
        """Return expected output shape."""
        if self._output_details:
            return tuple(self._output_details[0]["shape"])
        return (1, 1)

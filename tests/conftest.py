"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import MagicMock
import numpy as np


@pytest.fixture
def sample_vital():
    """Sample vital reading."""
    return {
        "timestamp": 1712973600000,
        "hr": 92.0,
        "bp_sys": 135.0,
        "bp_dia": 85.0,
        "o2_sat": 98.0,
        "temperature": 37.2,
        "quality": 95,
        "source": "simulator",
    }


@pytest.fixture
def sample_vital_unhealthy():
    """Unhealthy vital (sepsis indicators)."""
    return {
        "timestamp": 1712973600000,
        "hr": 125.0,  # Elevated
        "bp_sys": 95.0,  # Low
        "bp_dia": 55.0,  # Low
        "o2_sat": 88.0,  # Low
        "temperature": 39.5,  # Elevated
        "quality": 85,
        "source": "simulator",
    }


@pytest.fixture
def vital_sequence():
    """10 healthy vitals (100 seconds of data)."""
    vitals = []
    base_time = 1712973600000
    for i in range(10):
        vitals.append(
            {
                "timestamp": base_time + (i * 10000),
                "hr": 80.0 + np.random.normal(0, 3),
                "bp_sys": 120.0 + np.random.normal(0, 5),
                "bp_dia": 80.0 + np.random.normal(0, 4),
                "o2_sat": 97.0 + np.random.normal(0, 1),
                "temperature": 37.0 + np.random.normal(0, 0.3),
                "quality": 95,
                "source": "simulator",
            }
        )
    return vitals


@pytest.fixture
def mock_tflite_model():
    """Mock TFLite model."""
    mock_model = MagicMock()
    mock_model.allocate_tensors = MagicMock()
    mock_model.get_input_details = MagicMock(return_value=[{"index": 0, "shape": (1, 20)}])
    mock_model.get_output_details = MagicMock(return_value=[{"index": 0, "shape": (1, 1)}])
    mock_model.set_tensor = MagicMock()
    mock_model.invoke = MagicMock()
    mock_model.get_tensor = MagicMock(return_value=np.array([[0.75]]))
    return mock_model


@pytest.fixture
def mock_mqtt_client():
    """Mock MQTT client."""
    mock_client = MagicMock()
    mock_client.connect = MagicMock(return_value=(0, None))
    mock_client.subscribe = MagicMock(return_value=(0, 1))
    mock_client.publish = MagicMock(return_value=(0, 1))
    mock_client.disconnect = MagicMock()
    mock_client.is_connected = MagicMock(return_value=True)
    return mock_client

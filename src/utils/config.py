"""Configuration management."""

import os
from typing import Optional


class Config:
    """Configuration constants from environment variables."""
    
    # MQTT
    MQTT_BROKER: str = os.getenv("MQTT_BROKER", "localhost")
    MQTT_PORT: int = int(os.getenv("MQTT_PORT", "1883"))
    MQTT_TOPIC_VITALS: str = os.getenv(
        "MQTT_TOPIC_VITALS", "medtech/vitals/latest"
    )
    MQTT_TOPIC_PREDICTIONS: str = os.getenv(
        "MQTT_TOPIC_PREDICTIONS", "medtech/predictions/sepsis"
    )
    MQTT_QOS: int = int(os.getenv("MQTT_QOS", "1"))
    
    # Model & Inference
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/sepsis_model.tflite")
    INFERENCE_TIMEOUT_MS: int = int(os.getenv("INFERENCE_TIMEOUT_MS", "100"))
    
    # Vital Buffer
    BUFFER_SIZE: int = int(os.getenv("BUFFER_SIZE", "360"))  # 1 hour @ 10s
    VITAL_INTERVAL_S: int = int(os.getenv("VITAL_INTERVAL_S", "10"))
    
    # Scoring
    SEPSIS_THRESHOLD: float = float(os.getenv("SEPSIS_THRESHOLD", "0.5"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOGLEVEL", "INFO")
    LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] %(message)s"
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration."""
        assert cls.MQTT_PORT > 0, "MQTT_PORT must be > 0"
        assert cls.BUFFER_SIZE > 0, "BUFFER_SIZE must be > 0"
        assert 0 <= cls.SEPSIS_THRESHOLD <= 1, "SEPSIS_THRESHOLD must be 0-1"
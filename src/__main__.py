"""Entry point for edge analytics application."""

import argparse
import json
import signal
import threading
import time

from src.inference.sepsis_scorer import SepsisScorer
from src.inference.tflite_model import TFLiteModel
from src.inference.vital_buffer import VitalBuffer
from src.mqtt import mqtt_payload
from src.mqtt.mqtt_client import MQTTClient
from src.utils.config import Config
from src.utils.logger import setup_logger

# Scenario vital presets (v2 payload format)
_SCENARIOS = {
    "healthy": {
        "version": "2.0",
        "patient_id": "demo-healthy-001",
        "scenario": "healthy",
        "scenario_stage": "healthy",
        "hr": 80.0,
        "bp_sys": 120.0,
        "bp_dia": 80.0,
        "o2_sat": 97.0,
        "temperature": 37.0,
        "respiratory_rate": 16.0,
        "wbc": 7.5,
        "lactate": 0.8,
        "sirs_score": 0,
        "qsofa_score": 0,
        "sepsis_stage": "none",
        "sepsis_onset_ts": None,
        "quality": "good",
        "source": "simulator",
    },
    "sepsis": {
        "version": "2.0",
        "patient_id": "demo-sepsis-001",
        "scenario": "sepsis",
        "scenario_stage": "sepsis_onset",
        "hr": 125.0,
        "bp_sys": 95.0,
        "bp_dia": 55.0,
        "o2_sat": 88.0,
        "temperature": 39.5,
        "respiratory_rate": 26.0,
        "wbc": 14.5,
        "lactate": 3.2,
        "sirs_score": 3,
        "qsofa_score": 2,
        "sepsis_stage": "sepsis",
        "sepsis_onset_ts": None,
        "quality": "degraded",
        "source": "simulator",
    },
    "critical": {
        "version": "2.0",
        "patient_id": "demo-critical-001",
        "scenario": "critical",
        "scenario_stage": "septic_shock",
        "hr": 140.0,
        "bp_sys": 80.0,
        "bp_dia": 40.0,
        "o2_sat": 75.0,
        "temperature": 40.5,
        "respiratory_rate": 34.0,
        "wbc": 18.0,
        "lactate": 5.5,
        "sirs_score": 4,
        "qsofa_score": 3,
        "sepsis_stage": "septic_shock",
        "sepsis_onset_ts": None,
        "quality": "poor",
        "source": "simulator",
    },
}

_stop_event = threading.Event()


def _signal_handler(sig, frame):
    _stop_event.set()


def main():
    """Main entry point for edge analytics."""
    parser = argparse.ArgumentParser(description="MedTech Edge Analytics - Sepsis Detection")
    parser.add_argument(
        "--scenario",
        choices=["healthy", "sepsis", "critical"],
        help="Generate synthetic vitals for this scenario (no MQTT broker required)",
    )
    parser.add_argument(
        "--model-path",
        default=Config.MODEL_PATH,
        help="Path to TFLite model",
    )
    parser.add_argument(
        "--mqtt-broker",
        default=Config.MQTT_BROKER,
        help="MQTT broker host",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=Config.MQTT_PORT,
        help="MQTT broker port",
    )
    parser.add_argument(
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=Config.LOG_LEVEL,
        help="Logging level",
    )

    args = parser.parse_args()

    logger = setup_logger("medtech.analytics", args.loglevel)
    logger.info("MedTech Edge Analytics - Stage 1")
    logger.info("Scenario: %s", args.scenario)
    logger.info("Model path: %s", args.model_path)

    # Load TFLite model
    model = TFLiteModel(args.model_path)
    if not model.load():
        logger.warning(
            "Failed to load model from %s - running in degraded mode "
            "(predictions will be neutral 50%%)",
            args.model_path,
        )

    # Create pipeline components
    buffer = VitalBuffer(size=Config.BUFFER_SIZE)
    scorer = SepsisScorer(model)
    mqtt_client = MQTTClient(args.mqtt_broker, args.mqtt_port, client_id="medtech-edge")

    def on_vital_message(payload_str: str) -> None:
        """Handle incoming vital sign message."""
        try:
            logger.debug("Received vital payload: %s", payload_str)
            vital = mqtt_payload.parse_vital(payload_str)
            logger.debug("Parsed vital data: %s", vital)
            buffer.add_vital(vital)
            result = scorer.score(buffer)
            # Enrich prediction with traceability fields from the v2 payload
            result["patient_id"] = vital.get("patient_id", "unknown")
            result["vitals_version"] = vital.get("version", "2.0")
            result["vitals_timestamp"] = vital.get("timestamp")
            prediction_json = mqtt_payload.serialize_prediction(result)
            mqtt_client.publish(Config.MQTT_TOPIC_PREDICTIONS, prediction_json)
            logger.info(
                "Prediction: risk=%.1f%% (%s) latency=%.1fms",
                result["risk_score"],
                result["risk_level"],
                result["model_latency_ms"],
            )
        except ValueError as exc:
            logger.warning("Vital parsing error: %s", exc)
        except Exception as exc:
            logger.error("Processing error: %s", exc)

    mqtt_client.subscribe(Config.MQTT_TOPIC_VITALS, on_vital_message)

    if not mqtt_client.connect():
        logger.warning("MQTT connection failed - will retry in background")

    logger.info("Ready to receive vital sign messages on MQTT topic: %s", Config.MQTT_TOPIC_VITALS)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    scenario_vitals = _SCENARIOS.get(args.scenario) if args.scenario else None
    last_publish = 0.0

    logger.info("Running... Press Ctrl+C to stop")

    while not _stop_event.is_set():
        now = time.time()
        if scenario_vitals and (now - last_publish) >= Config.VITAL_INTERVAL_S:
            vital = dict(scenario_vitals)
            vital["timestamp"] = int(now * 1000)
            payload_str = json.dumps(vital)
            # In scenario mode process vitals directly to avoid duplicate predictions
            # if the broker echoes the publish back to the subscribed topic.
            on_vital_message(payload_str)
            last_publish = now

        # loop_start() drives the MQTT network loop in a background thread;
        # do NOT also call process() (client.loop()) from the main thread to
        # avoid racey callback execution.
        time.sleep(0.1)

    logger.info("Shutting down...")
    mqtt_client.disconnect()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()

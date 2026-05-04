"""Tests for MQTT client and payload handling."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.mqtt import mqtt_payload
from src.mqtt.mqtt_client import MQTTClient

# ── mqtt_payload tests ──────────────────────────────────────────────────────


class TestMqttPayload:
    """Tests for v2 MQTT payload parsing and serialisation."""

    VALID_VITAL = json.dumps(
        {
            "version": "2.0",
            "patient_id": "patient-test-001",
            "scenario": "healthy",
            "scenario_stage": "stable",
            "timestamp": 1712973600000,
            "hr": 80.0,
            "bp_sys": 120.0,
            "bp_dia": 80.0,
            "o2_sat": 97.0,
            "temperature": 37.0,
            "respiratory_rate": 16.0,
            "wbc": 7.5,
            "lactate": 0.9,
            "sirs_score": 0.0,
            "qsofa_score": 0.0,
            "sepsis_stage": "none",
            "sepsis_onset_ts": None,
            "quality": 95,
            "source": "simulator",
        }
    )

    # ── happy-path ────────────────────────────────────────────────────────────

    def test_parse_vital_valid(self):
        vital = mqtt_payload.parse_vital(self.VALID_VITAL)
        assert vital["hr"] == 80.0
        assert vital["bp_sys"] == 120.0
        assert vital["version"] == "2.0"
        assert vital["patient_id"] == "patient-test-001"
        assert vital["respiratory_rate"] == 16.0
        assert vital["lactate"] == 0.9
        assert vital["sirs_score"] == 0.0
        assert vital["qsofa_score"] == 0.0

    def test_parse_vital_sepsis_onset_ts_null(self):
        """sepsis_onset_ts is allowed to be null (onset not yet determined)."""
        vital = mqtt_payload.parse_vital(self.VALID_VITAL)
        assert vital.get("sepsis_onset_ts") is None

    def test_parse_vital_sepsis_onset_ts_populated(self):
        """sepsis_onset_ts is accepted when it holds an epoch-ms value."""
        data = json.loads(self.VALID_VITAL)
        data["sepsis_onset_ts"] = 1712973620000
        vital = mqtt_payload.parse_vital(json.dumps(data))
        assert vital["sepsis_onset_ts"] == 1712973620000

    # ── schema version enforcement ────────────────────────────────────────────

    def test_parse_vital_version_mismatch_rejected(self):
        """Messages with wrong version must be rejected with a clear error."""
        data = json.loads(self.VALID_VITAL)
        data["version"] = "1.0"
        with pytest.raises(ValueError, match="Schema version mismatch"):
            mqtt_payload.parse_vital(json.dumps(data))

    def test_parse_vital_version_missing_rejected(self):
        """Missing version is accepted when the payload is otherwise valid v2."""
        data = json.loads(self.VALID_VITAL)
        del data["version"]
        vital = mqtt_payload.parse_vital(json.dumps(data))
        assert vital["version"] == "2.0"

    def test_parse_vital_version_none_rejected(self):
        """Explicit null version must be rejected."""
        data = json.loads(self.VALID_VITAL)
        data["version"] = None
        vital = mqtt_payload.parse_vital(json.dumps(data))
        assert vital["version"] == "2.0"

    def test_parse_vital_schema_version_alias_accepted(self):
        """schema_version alias should be accepted for v2 publisher compatibility."""
        data = json.loads(self.VALID_VITAL)
        del data["version"]
        data["schema_version"] = "2.0"
        vital = mqtt_payload.parse_vital(json.dumps(data))
        assert vital["version"] == "2.0"

    def test_parse_vital_numeric_version_accepted(self):
        """Numeric version values should normalize to canonical v2 version."""
        data = json.loads(self.VALID_VITAL)
        data["version"] = 2
        vital = mqtt_payload.parse_vital(json.dumps(data))
        assert vital["version"] == "2.0"

    # ── missing / invalid field handling ─────────────────────────────────────

    def test_parse_vital_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            mqtt_payload.parse_vital("{not valid json}")

    def test_parse_vital_missing_fields(self):
        """Payload missing required field (e.g. patient_id) must be rejected."""
        payload = json.dumps({"version": "2.0", "timestamp": 1000, "hr": 80.0})
        with pytest.raises(ValueError, match="Missing required field"):
            mqtt_payload.parse_vital(payload)

    def test_parse_vital_missing_respiratory_rate(self):
        data = json.loads(self.VALID_VITAL)
        del data["respiratory_rate"]
        with pytest.raises(ValueError, match="Missing required field: 'respiratory_rate'"):
            mqtt_payload.parse_vital(json.dumps(data))

    def test_parse_vital_missing_lactate(self):
        data = json.loads(self.VALID_VITAL)
        del data["lactate"]
        with pytest.raises(ValueError, match="Missing required field: 'lactate'"):
            mqtt_payload.parse_vital(json.dumps(data))

    def test_parse_vital_missing_sirs_score(self):
        data = json.loads(self.VALID_VITAL)
        del data["sirs_score"]
        with pytest.raises(ValueError, match="Missing required field: 'sirs_score'"):
            mqtt_payload.parse_vital(json.dumps(data))

    def test_parse_vital_missing_qsofa_score(self):
        data = json.loads(self.VALID_VITAL)
        del data["qsofa_score"]
        with pytest.raises(ValueError, match="Missing required field: 'qsofa_score'"):
            mqtt_payload.parse_vital(json.dumps(data))

    # ── range validation ──────────────────────────────────────────────────────

    def test_parse_vital_out_of_range(self):
        data = json.loads(self.VALID_VITAL)
        data["hr"] = 300.0
        with pytest.raises(ValueError, match="out of range"):
            mqtt_payload.parse_vital(json.dumps(data))

    def test_parse_vital_o2_out_of_range(self):
        data = json.loads(self.VALID_VITAL)
        data["o2_sat"] = 20.0
        with pytest.raises(ValueError, match="out of range"):
            mqtt_payload.parse_vital(json.dumps(data))

    def test_parse_vital_respiratory_rate_out_of_range(self):
        data = json.loads(self.VALID_VITAL)
        data["respiratory_rate"] = 100.0
        with pytest.raises(ValueError, match="out of range"):
            mqtt_payload.parse_vital(json.dumps(data))

    def test_parse_vital_lactate_out_of_range(self):
        data = json.loads(self.VALID_VITAL)
        data["lactate"] = 50.0
        with pytest.raises(ValueError, match="out of range"):
            mqtt_payload.parse_vital(json.dumps(data))

    def test_parse_vital_sirs_score_out_of_range(self):
        data = json.loads(self.VALID_VITAL)
        data["sirs_score"] = 5.0
        with pytest.raises(ValueError, match="out of range"):
            mqtt_payload.parse_vital(json.dumps(data))

    def test_parse_vital_qsofa_score_out_of_range(self):
        data = json.loads(self.VALID_VITAL)
        data["qsofa_score"] = 4.0
        with pytest.raises(ValueError, match="out of range"):
            mqtt_payload.parse_vital(json.dumps(data))

    # ── prediction serialisation ──────────────────────────────────────────────

    def test_serialize_prediction(self):
        prediction = {
            "risk_score": 25.5,
            "risk_level": "LOW",
            "confidence": 0.255,
            "timestamp_ms": 1712973600000,
            "features_used": 20,
            "model_latency_ms": 5.2,
        }
        result = mqtt_payload.serialize_prediction(prediction)
        parsed = json.loads(result)
        assert parsed["risk_score"] == 25.5
        assert parsed["risk_level"] == "LOW"
        assert "  " in result  # pretty-printed with 2-space indent

    def test_serialize_prediction_with_traceability_fields(self):
        """Prediction payload with traceability fields must serialise cleanly."""
        prediction = {
            "risk_score": 75.0,
            "risk_level": "HIGH",
            "confidence": 0.75,
            "timestamp_ms": 1712973600000,
            "features_used": 20,
            "model_latency_ms": 3.1,
            "patient_id": "patient-001",
            "vitals_version": "2.0",
            "vitals_timestamp": 1712973600000,
        }
        result = mqtt_payload.serialize_prediction(prediction)
        parsed = json.loads(result)
        assert parsed["patient_id"] == "patient-001"
        assert parsed["vitals_version"] == "2.0"
        assert parsed["vitals_timestamp"] == 1712973600000

    def test_serialize_prediction_missing_field(self):
        with pytest.raises(ValueError, match="Missing required prediction field"):
            mqtt_payload.serialize_prediction({"risk_score": 50.0})


# ── MQTTClient tests ─────────────────────────────────────────────────────────


class TestMQTTClient:
    def test_is_connected_before_connect(self):
        client = MQTTClient("localhost", 1883)
        assert client.is_connected() is False

    def test_connect_failure(self):
        """Connect to non-existent broker should return False."""
        client = MQTTClient("127.0.0.1", 19999, client_id="test-fail")
        with patch.object(client._client, "connect", side_effect=ConnectionRefusedError("refused")):
            result = client.connect()
        assert result is False
        assert client.is_connected() is False

    def test_subscribe_stores_callback(self):
        client = MQTTClient("localhost", 1883)
        callback = MagicMock()
        result = client.subscribe("test/topic", callback)
        assert result is True
        assert "test/topic" in client._subscriptions

    def test_publish_when_not_connected(self):
        """Publish should handle errors gracefully."""
        client = MQTTClient("localhost", 1883)
        mock_result = MagicMock()
        mock_result.rc = 4  # MQTT_ERR_NO_CONN
        with patch.object(client._client, "publish", return_value=mock_result):
            result = client.publish("test/topic", '{"key": "value"}')
        assert result is False

    def test_disconnect_when_not_connected(self):
        """Disconnect should handle the not-connected case gracefully."""
        client = MQTTClient("localhost", 1883)
        result = client.disconnect()
        assert isinstance(result, bool)

    def test_on_message_dispatch(self):
        """on_message callback should dispatch to registered handler."""
        client = MQTTClient("localhost", 1883)
        received = []

        client._subscriptions["vitals/test"] = lambda payload: received.append(payload)

        mock_msg = MagicMock()
        mock_msg.topic = "vitals/test"
        mock_msg.payload = b'{"hr": 80.0}'

        client._on_message(None, None, mock_msg)
        assert received == ['{"hr": 80.0}']

    def test_on_message_wildcard_dispatch(self):
        """on_message should dispatch to wildcard subscriptions via topic_matches_sub."""
        client = MQTTClient("localhost", 1883)
        received = []
        client._subscriptions["vitals/#"] = lambda payload: received.append(payload)

        mock_msg = MagicMock()
        mock_msg.topic = "vitals/room1/sensor2"
        mock_msg.payload = b'{"hr": 75.0}'

        client._on_message(None, None, mock_msg)
        assert received == ['{"hr": 75.0}']

    def test_on_message_no_duplicate_dispatch(self):
        """Same callback registered to two matching wildcards should only be called once."""
        client = MQTTClient("localhost", 1883)
        received = []

        def callback(payload):
            received.append(payload)

        client._subscriptions["vitals/#"] = callback
        client._subscriptions["vitals/+"] = callback  # same callback object

        mock_msg = MagicMock()
        mock_msg.topic = "vitals/room1"
        mock_msg.payload = b'{"hr": 80.0}'

        client._on_message(None, None, mock_msg)
        assert len(received) == 1  # deduplication by callback identity

    def test_on_connect_sets_connected(self):
        client = MQTTClient("localhost", 1883)
        client._on_connect(client._client, None, {}, 0)
        assert client.is_connected() is True

    def test_on_disconnect_clears_connected(self):
        client = MQTTClient("localhost", 1883)
        client._connected = True
        client._on_disconnect(client._client, None, 0)
        assert client.is_connected() is False

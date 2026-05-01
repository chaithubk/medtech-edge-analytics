"""Tests for MQTT client and payload handling."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.mqtt import mqtt_payload
from src.mqtt.mqtt_client import MQTTClient

# ── mqtt_payload tests ──────────────────────────────────────────────────────


class TestMqttPayload:
    VALID_VITAL = json.dumps(
        {
            "timestamp": 1712973600000,
            "hr": 80.0,
            "bp_sys": 120.0,
            "bp_dia": 80.0,
            "o2_sat": 97.0,
            "temperature": 37.0,
            "quality": 95,
            "source": "simulator",
        }
    )

    def test_parse_vital_valid(self):
        vital = mqtt_payload.parse_vital(self.VALID_VITAL)
        assert vital["hr"] == 80.0
        assert vital["bp_sys"] == 120.0

    def test_parse_vital_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            mqtt_payload.parse_vital("{not valid json}")

    def test_parse_vital_missing_fields(self):
        payload = json.dumps({"timestamp": 1000, "hr": 80.0})
        with pytest.raises(ValueError, match="Missing required field"):
            mqtt_payload.parse_vital(payload)

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

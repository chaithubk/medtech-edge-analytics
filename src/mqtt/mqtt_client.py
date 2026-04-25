"""MQTT client for vital subscription and prediction publishing."""

import time
from typing import Callable, Dict

import paho.mqtt.client as mqtt

from src.utils.logger import get_logger

logger = get_logger(__name__)

_CONNECT_TIMEOUT_S = 10
_RECONNECT_DELAY_MAX_S = 60


class MQTTClient:
    """MQTT client for vital subscription and prediction publishing."""

    def __init__(self, broker_host: str, broker_port: int, client_id: str = "") -> None:
        """Initialize MQTT client with connection parameters.

        Args:
            broker_host: Hostname or IP of the MQTT broker.
            broker_port: TCP port of the MQTT broker.
            client_id: Optional MQTT client identifier.
        """
        self._host = broker_host
        self._port = broker_port
        self._client_id = client_id
        self._connected = False
        self._subscriptions: Dict[str, Callable] = {}

        self._client = mqtt.Client(client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        self._client.reconnect_delay_set(min_delay=1, max_delay=_RECONNECT_DELAY_MAX_S)

    def connect(self) -> bool:
        """Connect to the MQTT broker.

        Returns:
            True if connection was initiated successfully, False otherwise.
        """
        try:
            self._client.connect(self._host, self._port, keepalive=60)
            self._client.loop_start()
            deadline = time.time() + _CONNECT_TIMEOUT_S
            while not self._connected and time.time() < deadline:
                time.sleep(0.05)
            if self._connected:
                logger.info("Connected to MQTT broker %s:%d", self._host, self._port)
                return True
            # Timed out - stop the background loop to avoid a leaked thread
            logger.error("Timed out waiting for MQTT connection")
            self._client.loop_stop()
            return False
        except Exception as exc:
            logger.error("MQTT connect failed: %s", exc)
            try:
                self._client.loop_stop()
            except Exception:
                pass
            return False

    def disconnect(self) -> bool:
        """Gracefully disconnect from the MQTT broker.

        Returns:
            True if disconnected successfully, False otherwise.
        """
        try:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
            logger.info("Disconnected from MQTT broker")
            return True
        except Exception as exc:
            logger.warning("MQTT disconnect warning: %s", exc)
            return False

    def subscribe(self, topic: str, callback: Callable) -> bool:
        """Subscribe to an MQTT topic with a message callback.

        Args:
            topic: MQTT topic string (supports wildcards).
            callback: Function called with the decoded message payload string.

        Returns:
            True if subscription was registered, False otherwise.
        """
        try:
            self._subscriptions[topic] = callback
            if self._connected:
                result, _ = self._client.subscribe(topic, qos=1)
                if result == mqtt.MQTT_ERR_SUCCESS:
                    logger.info("Subscribed to topic: %s", topic)
                    return True
                logger.warning("Subscription failed for topic: %s (rc=%d)", topic, result)
                return False
            logger.debug("Stored subscription for topic: %s (not yet connected)", topic)
            return True
        except Exception as exc:
            logger.error("Subscribe error: %s", exc)
            return False

    def publish(self, topic: str, payload: str) -> bool:
        """Publish a JSON payload to an MQTT topic.

        Args:
            topic: MQTT topic string.
            payload: JSON-encoded string to publish.

        Returns:
            True if publish succeeded, False otherwise.
        """
        try:
            result = self._client.publish(topic, payload, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug("Published to %s", topic)
                return True
            logger.warning("Publish failed to %s (rc=%d)", topic, result.rc)
            return False
        except Exception as exc:
            logger.warning("Publish error: %s", exc)
            return False

    def is_connected(self) -> bool:
        """Return True if currently connected to the broker."""
        return self._connected

    def process(self, timeout_ms: int = 100) -> None:
        """Process pending MQTT network events (non-blocking).

        Args:
            timeout_ms: Maximum time to wait for network activity in milliseconds.
        """
        try:
            self._client.loop(timeout=timeout_ms / 1000.0)
        except Exception as exc:
            logger.warning("MQTT loop error: %s", exc)

    def _on_connect(self, client: mqtt.Client, userdata: object, flags: dict, rc: int) -> None:
        """Handle successful connection to broker."""
        if rc == 0:
            self._connected = True
            logger.info("MQTT on_connect: connected (rc=0)")
            for topic in self._subscriptions:
                client.subscribe(topic, qos=1)
                logger.debug("Re-subscribed to %s", topic)
        else:
            logger.error("MQTT on_connect: failed with rc=%d", rc)

    def _on_disconnect(self, client: mqtt.Client, userdata: object, rc: int) -> None:
        """Handle disconnection; auto-reconnect is handled by paho."""
        self._connected = False
        if rc != 0:
            logger.warning("Unexpected MQTT disconnect (rc=%d), auto-reconnect enabled", rc)
        else:
            logger.info("MQTT disconnected cleanly")

    def _on_message(self, client: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage) -> None:
        """Dispatch incoming message to all callbacks whose subscription matches the topic.

        Supports MQTT wildcard patterns ('+' and '#') via paho's topic_matches_sub().
        """
        try:
            payload_str = msg.payload.decode("utf-8")
            matched = False
            seen_callbacks: set = set()
            for subscription_topic, callback in self._subscriptions.items():
                if mqtt.topic_matches_sub(subscription_topic, msg.topic):
                    callback_id = id(callback)
                    if callback_id not in seen_callbacks:
                        seen_callbacks.add(callback_id)
                        matched = True
                        callback(payload_str)
            if not matched:
                logger.debug("No callback for topic: %s", msg.topic)
        except Exception as exc:
            logger.error("Error in message callback for %s: %s", msg.topic, exc)

"""Unit tests for runtime contract schema path resolution.

Covers:
- Path resolution uses MEDTECH_VITALS_SCHEMA env var when set.
- Path resolution falls back to the default rootfs path when env var is absent.
- resolve_schema_path() raises FileNotFoundError when the file does not exist.
- resolve_schema_path() raises FileNotFoundError when the file is unreadable.
- The service's main() hard-fails (sys.exit(1)) when schema is missing.
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.utils.schema_loader import _DEFAULT_SCHEMA_PATH, _ENV_VAR, resolve_schema_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VENDORED_SCHEMA = pathlib.Path(__file__).parent.parent / "contracts" / "vitals" / "v2.0.json"


# ---------------------------------------------------------------------------
# resolve_schema_path — env var resolution
# ---------------------------------------------------------------------------


class TestResolveSchemaPathEnvVar:
    """resolve_schema_path honours MEDTECH_VITALS_SCHEMA when set."""

    def test_env_var_path_is_used(self, tmp_path, monkeypatch):
        """When MEDTECH_VITALS_SCHEMA points to an existing file that path is returned."""
        schema_file = tmp_path / "vitals.json"
        schema_file.write_text("{}")
        monkeypatch.setenv(_ENV_VAR, str(schema_file))

        result = resolve_schema_path()

        assert result == schema_file

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        """MEDTECH_VITALS_SCHEMA takes priority over the default rootfs path."""
        schema_file = tmp_path / "custom_schema.json"
        schema_file.write_text("{}")
        monkeypatch.setenv(_ENV_VAR, str(schema_file))

        result = resolve_schema_path()

        assert str(result) != _DEFAULT_SCHEMA_PATH
        assert result == schema_file

    def test_empty_env_var_falls_back_to_default(self, monkeypatch, tmp_path):
        """An empty MEDTECH_VITALS_SCHEMA string is treated as unset."""
        # Point default path to a real file so the fallback succeeds.
        default_file = tmp_path / "current.json"
        default_file.write_text("{}")
        monkeypatch.setenv(_ENV_VAR, "")

        with patch("src.utils.schema_loader._DEFAULT_SCHEMA_PATH", str(default_file)):
            result = resolve_schema_path()

        assert result == default_file


# ---------------------------------------------------------------------------
# resolve_schema_path — default path fallback
# ---------------------------------------------------------------------------


class TestResolveSchemaPathDefault:
    """resolve_schema_path uses the default path when env var is absent."""

    def test_default_path_returned_when_env_absent(self, tmp_path, monkeypatch):
        """When env var is not set the default rootfs path is used."""
        monkeypatch.delenv(_ENV_VAR, raising=False)
        default_file = tmp_path / "current.json"
        default_file.write_text("{}")

        with patch("src.utils.schema_loader._DEFAULT_SCHEMA_PATH", str(default_file)):
            result = resolve_schema_path()

        assert result == default_file

    def test_vendored_schema_accessible_via_env_var(self, monkeypatch):
        """The vendored CI schema is loadable when injected via env var."""
        assert _VENDORED_SCHEMA.exists(), "Vendored schema must be present for CI"
        monkeypatch.setenv(_ENV_VAR, str(_VENDORED_SCHEMA))

        result = resolve_schema_path()

        assert result == _VENDORED_SCHEMA


# ---------------------------------------------------------------------------
# resolve_schema_path — hard-fail on missing/unreadable schema
# ---------------------------------------------------------------------------


class TestResolveSchemaPathFailure:
    """resolve_schema_path raises FileNotFoundError for invalid paths."""

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        """A path that does not exist must raise FileNotFoundError."""
        missing = tmp_path / "does_not_exist.json"
        monkeypatch.setenv(_ENV_VAR, str(missing))

        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_schema_path()

    def test_unreadable_file_raises(self, tmp_path, monkeypatch):
        """A path that exists but is not readable must raise PermissionError."""
        import os

        schema_file = tmp_path / "no_read.json"
        schema_file.write_text("{}")
        monkeypatch.setenv(_ENV_VAR, str(schema_file))

        # Mock os.access to return False for this file
        monkeypatch.setattr(os, "access", lambda path, mode: False)

        with pytest.raises(PermissionError, match="not readable"):
            resolve_schema_path()

    def test_default_missing_raises(self, monkeypatch):
        """When the default path does not exist FileNotFoundError is raised."""
        monkeypatch.delenv(_ENV_VAR, raising=False)

        with patch(
            "src.utils.schema_loader._DEFAULT_SCHEMA_PATH",
            "/nonexistent/path/current.json",
        ):
            with pytest.raises(FileNotFoundError):
                resolve_schema_path()


# ---------------------------------------------------------------------------
# main() hard-fail integration
# ---------------------------------------------------------------------------


class TestMainHardFailOnMissingSchema:
    """The service entry-point must call sys.exit(1) when schema is missing."""

    def test_main_exits_nonzero_when_schema_missing(self, monkeypatch):
        """main() must exit with code 1 when resolve_schema_path raises."""
        monkeypatch.setenv(_ENV_VAR, "/nonexistent/schema.json")

        # Patch sys.argv so argparse does not read real CLI args.
        monkeypatch.setattr(sys, "argv", ["medtech-edge-analytics"])

        with pytest.raises(SystemExit) as exc_info:
            from src.__main__ import main  # noqa: PLC0415

            main()

        assert exc_info.value.code == 1

    def test_main_does_not_exit_when_schema_present(self, tmp_path, monkeypatch):
        """main() must NOT call sys.exit when the schema file exists."""
        schema_file = tmp_path / "current.json"
        schema_file.write_text("{}")
        monkeypatch.setenv(_ENV_VAR, str(schema_file))

        # Patch sys.argv to select scenario mode so no MQTT connection is needed.
        monkeypatch.setattr(sys, "argv", ["medtech-edge-analytics", "--scenario", "healthy"])

        # Patch the blocking loop so the test returns immediately.
        with (
            patch("src.__main__.TFLiteModel") as mock_model_cls,
            patch("src.__main__.MQTTClient") as mock_mqtt_cls,
            patch("src.__main__._stop_event") as mock_event,
        ):
            mock_model = MagicMock()
            mock_model.load.return_value = True
            mock_model_cls.return_value = mock_model

            mock_mqtt = MagicMock()
            mock_mqtt.connect.return_value = True
            mock_mqtt_cls.return_value = mock_mqtt

            # Make the main loop exit immediately.
            mock_event.is_set.return_value = True

            # Should complete without SystemExit.
            from src.__main__ import main  # noqa: PLC0415

            main()

"""Contract pin metadata consistency tests for edge-analytics."""

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
PIN_PATH = REPO_ROOT / "contracts" / "vitals" / "contract-pin.json"
SCHEMA_PATH = REPO_ROOT / "contracts" / "vitals" / "vitals.schema.json"
MANIFEST_PATH = REPO_ROOT / "contracts" / "vitals" / "vitals.schema-manifest.yml"
VERSION_PATH = REPO_ROOT / "contracts" / "VITALS_CONTRACT_VERSION.txt"


def test_contract_pin_file_exists_and_has_required_fields() -> None:
    assert PIN_PATH.exists(), f"Missing contract pin metadata: {PIN_PATH}"
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))

    required = {
        "contract_repo",
        "tag",
        "commit_sha",
        "schema_path",
        "local_schema",
        "compatibility",
    }
    assert required.issubset(pin.keys())


def test_contract_pin_paths_and_tag_are_consistent() -> None:
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    assert pin["local_schema"] == "contracts/vitals/vitals.schema.json"
    assert pin["schema_path"] == "schemas/vitals/vitals.schema.json"
    assert VERSION_PATH.read_text(encoding="utf-8").strip() == pin["tag"]


def test_contract_pin_matches_manifest_version() -> None:
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    current_version = str(manifest.get("current_version", "")).strip()
    assert current_version == pin["tag"].lstrip("v")


def test_contract_pin_schema_file_exists() -> None:
    assert SCHEMA_PATH.exists(), f"Missing schema file referenced by contract pin: {SCHEMA_PATH}"

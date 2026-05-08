#!/usr/bin/env python3
"""
Vendor the vitals contract schema from the upstream repo at a given tag.

- Downloads the schema for the specified tag from the upstream contract repo.
- Writes it to contracts/vitals/current.json (overwriting if exists).
- Updates contracts/VITALS_CONTRACT_VERSION.txt with the tag.

Usage:
    python tools/vendor_telemetry_contract.py <tag>

Example:
    python tools/vendor_telemetry_contract.py v2.1.0
    python tools/vendor_telemetry_contract.py 2.1.0
"""

import os
import pathlib
import sys

import requests

CONTRACT_REPO = "chaithubk/medtech-telemetry-contract"
SCHEMA_PATH = "schemas/vitals/v2.0.json"  # Upstream path is always v2.0.json for now
VENDORED_SCHEMA = pathlib.Path("contracts/vitals/current.json")
PIN_FILE = pathlib.Path("contracts/VITALS_CONTRACT_VERSION.txt")


def fetch_schema(tag: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{CONTRACT_REPO}/{tag}/{SCHEMA_PATH}"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch schema: {url} (status {resp.status_code})")
    return resp.content


def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/vendor_telemetry_contract.py <tag>", file=sys.stderr)
        sys.exit(1)
    tag = sys.argv[1]
    os.makedirs(VENDORED_SCHEMA.parent, exist_ok=True)
    schema_bytes = fetch_schema(tag)
    VENDORED_SCHEMA.write_bytes(schema_bytes)
    PIN_FILE.write_text(tag + "\n")
    print(f"Vendored schema for tag {tag} -> {VENDORED_SCHEMA}")
    print(f"Updated pin file: {PIN_FILE}")


if __name__ == "__main__":
    main()

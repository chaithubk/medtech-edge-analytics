#!/usr/bin/env python3
"""Write normalized contract pin metadata used by CI and runtime checks."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re


def read_compatibility_class(manifest_path: pathlib.Path) -> str:
    if not manifest_path.exists():
        return "unknown"
    content = manifest_path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^compatibility_class:\s*\"?([^\"\n#]+)", content)
    if not match:
        return "unknown"
    return match.group(1).strip().lower()


parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
parser.add_argument("--commit", required=True)
parser.add_argument("--manifest", default="contracts/vitals/vitals.schema-manifest.yml")
parser.add_argument("--output", default="contracts/vitals/contract-pin.json")
args = parser.parse_args()

manifest_path = pathlib.Path(args.manifest)
compat_class = read_compatibility_class(manifest_path)

metadata = {
    "contract_repo": "chaithubk/medtech-telemetry-contract",
    "tag": args.tag,
    "commit_sha": args.commit,
    "schema_path": "schemas/vitals/vitals.schema.json",
    "resolved_schema_path": "schemas/vitals/vitals.schema.json",
    "local_schema": "contracts/vitals/vitals.schema.json",
    "synced_at_utc": dt.datetime.now(dt.UTC)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z"),
    "compatibility": {
        "classification": compat_class,
        "breaking": compat_class in {"breaking", "unknown"},
        "source": "manifest",
    },
    "application_code_changes_required": compat_class in {"breaking", "unknown"},
}

output_path = pathlib.Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {output_path}")

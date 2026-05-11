#!/usr/bin/env python3
"""Read compatibility_class from the vendored manifest without YAML deps."""

from __future__ import annotations

import argparse
import pathlib
import re

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="contracts/vitals/vitals.schema-manifest.yml")
args = parser.parse_args()

manifest_path = pathlib.Path(args.manifest)
if not manifest_path.exists():
    print("UNKNOWN")
    raise SystemExit(0)

content = manifest_path.read_text(encoding="utf-8")
match = re.search(r"(?m)^compatibility_class:\s*\"?([^\"\n#]+)", content)
if not match:
    print("UNKNOWN")
    raise SystemExit(0)

print(match.group(1).strip())

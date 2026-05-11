#!/usr/bin/env python3
"""Stamp pinned_tag and pinned_commit values into the vendored manifest."""

from __future__ import annotations

import argparse
import pathlib
import re

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="contracts/vitals/vitals.schema-manifest.yml")
parser.add_argument("--tag", required=True)
parser.add_argument("--commit", required=True)
args = parser.parse_args()

manifest_path = pathlib.Path(args.manifest)
if not manifest_path.exists():
    raise SystemExit(0)

content = manifest_path.read_text(encoding="utf-8")

if "pinned_commit:" in content:
    content = re.sub(r"(?m)^pinned_commit:\s*.*$", f'pinned_commit: "{args.commit}"', content)
else:
    if not content.endswith("\n"):
        content += "\n"
    content += f'pinned_commit: "{args.commit}"\n'

if "pinned_tag:" in content:
    content = re.sub(r"(?m)^pinned_tag:\s*.*$", f"pinned_tag: {args.tag}", content)
else:
    if not content.endswith("\n"):
        content += "\n"
    content += f"pinned_tag: {args.tag}\n"

manifest_path.write_text(content, encoding="utf-8")
print(f"Stamped commit {args.commit} and tag {args.tag} into manifest.")

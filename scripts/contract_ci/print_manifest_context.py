#!/usr/bin/env python3
"""Print helpful informational fields from the vendored manifest."""

from __future__ import annotations

from pathlib import Path
import re

manifest_path = Path("contracts/vitals/vitals.schema-manifest.yml")
if not manifest_path.exists():
    raise SystemExit(0)

content = manifest_path.read_text(encoding="utf-8")

def get_value(key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*\"?([^\"\n#]+)", content)
    if not match:
        return "unknown"
    return match.group(1).strip()

print(f"  Pinned commit : {get_value('pinned_commit')}")
print(f"  Compat class  : {get_value('compatibility_class')}")
print(f"  Vendored at   : {get_value('vendored_at')}")

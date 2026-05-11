#!/usr/bin/env python3
"""Read pinned contract tag from contract-pin.json with version-file fallback."""

from __future__ import annotations

import json
from pathlib import Path

pin_file = Path("contracts/vitals/contract-pin.json")
if pin_file.exists():
    try:
        data = json.loads(pin_file.read_text(encoding="utf-8"))
        print(str(data.get("tag") or "").strip())
        raise SystemExit(0)
    except Exception:
        pass

version_file = Path("contracts/VITALS_CONTRACT_VERSION.txt")
if version_file.exists():
    print(version_file.read_text(encoding="utf-8").strip())
else:
    print("")

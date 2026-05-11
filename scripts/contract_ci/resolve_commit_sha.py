#!/usr/bin/env python3
"""Resolve the immutable commit SHA for a given contract tag."""

from __future__ import annotations

import argparse
import json
import urllib.request

API_BASE = "https://api.github.com/repos/chaithubk/medtech-telemetry-contract"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def fetch(url: str):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
args = parser.parse_args()

try:
    ref = fetch(f"{API_BASE}/git/ref/tags/{args.tag}")
    obj = ref.get("object", {}) if isinstance(ref, dict) else {}
    sha = str(obj.get("sha") or "").strip()
    typ = str(obj.get("type") or "").strip()

    # Annotated tag: dereference to the commit object.
    if typ == "tag" and sha:
        tag_obj = fetch(f"{API_BASE}/git/tags/{sha}")
        if isinstance(tag_obj, dict):
            sha = str((tag_obj.get("object") or {}).get("sha") or sha).strip()

    print(sha)
except Exception:
    print("")

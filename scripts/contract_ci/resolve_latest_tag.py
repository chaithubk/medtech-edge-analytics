#!/usr/bin/env python3
"""Resolve the latest release/tag from the telemetry contract repository."""

from __future__ import annotations

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


try:
    release = fetch(f"{API_BASE}/releases/latest")
    tag_name = str(release.get("tag_name") or "").strip()
    if tag_name and tag_name != "null":
        print(tag_name)
    else:
        tags = fetch(f"{API_BASE}/tags")
        if isinstance(tags, list) and tags:
            print(str(tags[0].get("name") or "").strip())
        else:
            print("")
except Exception:
    print("")

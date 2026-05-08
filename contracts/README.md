# Telemetry Contract

This directory contains the vendored telemetry contract schema used by
`medtech-edge-analytics` to validate inbound MQTT vital-sign payloads.

## Source of Truth

The canonical contract lives in the central contract repository:

> **[chaithubk/medtech-telemetry-contract](https://github.com/chaithubk/medtech-telemetry-contract)**

The copy in this directory is pinned to the tag recorded in
`VITALS_CONTRACT_VERSION.txt` (currently **v2.0.0**) and vendored here for
offline / Yocto build reproducibility.

## Directory Layout

```
contracts/
├── VITALS_CONTRACT_VERSION.txt   # Pinned tag from the contract repo
├── README.md                     # This file
└── vitals/
   └── current.json                 # Vendored JSON Schema (draft-07)
```

## How the Schema Is Used

`tests/test_contract_schema_v2.py` validates every fixture payload against
`contracts/vitals/current.json` using the `jsonschema` library (dev dependency).
This ensures any payload drift is caught immediately as a failing CI test.

The runtime parser (`src/mqtt/mqtt_payload.py`) enforces the v2 contract
programmatically (required fields, version sentinel, numeric ranges).

### Runtime schema path resolution

At runtime the service resolves the schema file through the following priority
chain (implemented in `src/utils/schema_loader.py`):

1. `MEDTECH_VITALS_SCHEMA` environment variable (if set and non-empty).
2. Default Yocto rootfs path: `/usr/share/medtech/contracts/vitals/current.json`.

If the resolved file is **missing or unreadable the service hard-fails** (exits
with code `1`).  The system is not backward-compatible and is not intended to
run without a valid contract file.


The vendored copy at `contracts/vitals/current.json` is used in tests and CI by
setting the env var:

```bash
MEDTECH_VITALS_SCHEMA=contracts/vitals/current.json pytest tests/
```

On a Yocto device the `medtech-telemetry-contract` package installs the schema
to `/usr/share/medtech/contracts/vitals/current.json` (a symlink to the pinned
version file), so no env var override is needed in production.

## Update Procedure

When the upstream contract repo publishes a new tag:

1. **Detect** – the `Contract Drift Check` GitHub Actions workflow runs daily and
   fails with a clear message if a newer tag is available in
   `chaithubk/medtech-telemetry-contract`.

2. **Vendor** – trigger the `Vendor Telemetry Contract` workflow
   (`workflow_dispatch`) and provide the new tag as input. The workflow will:
   - Download the schema at the new tag
   - Update `contracts/vitals/current.json` and `contracts/VITALS_CONTRACT_VERSION.txt`
   - Open a PR automatically

3. **Review** – the PR will include the schema diff.  CI validates that all
   fixtures still pass against the new schema.  Fix any fixture or parser
   mismatches, then merge.

## Policy

> The payload emitted by `medtech-vitals-publisher` and consumed by
> `medtech-edge-analytics` **must** validate against the schema in this
> directory at the currently pinned tag.  Drift is a CI failure.

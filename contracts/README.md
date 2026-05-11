# Telemetry Contract

This directory contains the vendored telemetry contract schema used by
`medtech-edge-analytics` to validate inbound MQTT vital-sign payloads.

## Source of Truth

The canonical contract lives in the central contract repository:

> **[chaithubk/medtech-telemetry-contract](https://github.com/chaithubk/medtech-telemetry-contract)**

The copy in this directory is pinned to an immutable contract revision and
vendored for offline / Yocto build reproducibility.

Active pinned revision:

- Tag: `vitals/contract-pin.json` -> `tag`
- Commit SHA: `vitals/contract-pin.json` -> `commit_sha`
- Compatibility class: `vitals/contract-pin.json` -> `compatibility.classification`

## Directory Layout

```
contracts/
├── VITALS_CONTRACT_VERSION.txt     # Pinned tag from contract repo
├── README.md                       # This file
└── vitals/
   ├── contract-pin.json           # Structured pin metadata (canonical)
   ├── vitals.schema.json          # Canonical vendored JSON Schema
   └── vitals.schema-manifest.yml  # Governance metadata (compat, commit pin)
```

## How the Schema Is Used

`tests/test_contract_schema_v2.py` validates every fixture payload against
`contracts/vitals/vitals.schema.json` using the `jsonschema` library.
This ensures any payload drift is caught immediately as a failing CI test.

The runtime parser (`src/mqtt/mqtt_payload.py`) enforces safe compatibility:

- SemVer compatibility check (accepts `2.x.x`, rejects MAJOR mismatch)
- Unknown fields stripped with warning (forward compatible MINOR/PATCH growth)
- Required-field checks and numeric/type validation

### Runtime schema path resolution

At runtime the service resolves the schema file through the following priority
chain (implemented in `src/utils/schema_loader.py`):

1. `MEDTECH_VITALS_SCHEMA` environment variable (if set and non-empty).
2. Default Yocto rootfs path: `/usr/share/medtech/contracts/vitals/vitals.schema.json`.

If the resolved file is **missing or unreadable the service hard-fails** (exits
with code `1`).  The system is not backward-compatible and is not intended to
run without a valid contract file.

The vendored copy at `contracts/vitals/vitals.schema.json` is used in tests and CI by
setting the env var:

```bash
MEDTECH_VITALS_SCHEMA=contracts/vitals/vitals.schema.json pytest tests/
```

On a Yocto device the `medtech-telemetry-contract` package installs the schema
to `/usr/share/medtech/contracts/vitals/vitals.schema.json`, so no env var
override is needed in production.

## Update Procedure

When the upstream contract repo publishes a new tag:

1. **Detect** – the `Contract Drift Check` GitHub Actions workflow runs daily and
   fails with a clear message if a newer tag is available in
   `chaithubk/medtech-telemetry-contract`.

2. **Vendor** – trigger the `Vendor Telemetry Contract` workflow
   (`workflow_dispatch`) and provide the new tag as input. The workflow will:
   - Download canonical schema from `schemas/vitals/vitals.schema.json`
   - Download governance metadata from `schemas/vitals/vitals.schema-manifest.yml`
   - Update `contracts/vitals/contract-pin.json`
   - Update `contracts/VITALS_CONTRACT_VERSION.txt` (legacy compatibility)
   - Stamp exact commit SHA (`pinned_commit`) in vendored manifest
   - Open a PR automatically

3. **Review** – the PR will include the schema diff.  CI validates that all
   fixtures still pass against the new schema.  Fix any fixture or parser
   mismatches, then merge.

## Compatibility Policy

Compatibility class is read from vendored manifest and drives merge policy:

- `PATCH`: tests must pass; may auto-merge.
- `MINOR`: manual review required.
- `BREAKING`: manual review + migration checklist required.

See `docs/contract-pinning.md` for full migration and upgrade guidance.

## Policy

> The payload emitted by `medtech-vitals-publisher` and consumed by
> `medtech-edge-analytics` **must** validate against the schema in this
> directory at the currently pinned tag.  Drift is a CI failure.

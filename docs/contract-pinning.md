# Telemetry Contract Pinning

This repository consumes telemetry as an external interface and pins a specific
contract revision.

## What Is Pinned

The active contract revision is identified by:

- Tag: `contracts/VITALS_CONTRACT_VERSION.txt`
- Commit SHA: `contracts/vitals/vitals.schema-manifest.yml` -> `pinned_commit`
- Canonical schema path in contract repo: `schemas/vitals/vitals.schema.json`

Vendored artifacts in this repository:

- `contracts/vitals/vitals.schema.json`
- `contracts/vitals/vitals.schema-manifest.yml`

## Runtime Behavior

Runtime validation is version-aware and safe across revisions:

1. Read `payload.version` (SemVer expected).
2. Accept payloads compatible with pinned MAJOR version (`2.x.x`).
3. Reject payloads with different MAJOR version as BREAKING.
4. Strip unknown fields from newer MINOR/PATCH revisions and continue.
5. Enforce all required fields and type/range checks from pinned contract.

If the contract schema file is missing or unreadable at startup, the service
exits with a fatal error.

## Compatibility Classes

Compatibility class comes from the vendored manifest and governs merge policy:

- `PATCH`: tests must pass; may auto-merge.
- `MINOR`: human review required.
- `BREAKING`: human review + migration checklist required.

## How To Upgrade

1. Run workflow `Vendor Telemetry Contract` and choose target tag.
2. The workflow vendors canonical schema + manifest and pins commit SHA.
3. Review generated PR labels:
   - `contract-update`
   - `compat:patch|minor|breaking`
4. Run/verify CI, especially contract tests:
   - `tests/test_contract_schema_v2.py`
   - `tests/test_contract_evolution.py`
   - `tests/test_contract_compat.py`
5. For `BREAKING`, complete migration checklist before merge.

## Safe vs Breaking Upgrades

Safe (usually no code changes):

- PATCH fix in schema validation logic.
- MINOR adding optional fields.
- MINOR adding enum members not consumed as strict assumptions.

Potentially breaking (code changes likely required):

- Required field added/removed/renamed.
- Field type changes.
- Contract MAJOR version bump.
- Enum changes that invalidate runtime assumptions.

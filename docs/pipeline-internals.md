# Pipeline Internals: Training to Deployable Edge Model

This document is the source of truth for how the CI pipeline produces and versions
the model consumed by edge and Yocto builds.

## Scope

- Training data source: synthetic FHIR records from Synthea sepsis module
- Processing: FHIR flattening into tabular ML dataset
- Output artifact: `models/imx8-compatible-sepsis.tflite`
- Delivery: model committed to git and version-tagged by CI

## End-to-End Flow

1. Generate synthetic FHIR data with Synthea into `data/raw_fhir/fhir/`.
2. Flatten FHIR records into `data/processed/dataset.csv` via `src/flatten_fhir.py`.
3. Train and validate model in `src/train_and_convert.py`.
4. Quantize/export TensorFlow Lite model to `models/imx8-compatible-sepsis.tflite`.
5. Publish reports and metadata (`pipeline_report.md`, `model_metadata.json`, logs).
6. If model changed, CI commits updated model and creates model version tag.

## CI Triggering and Control

- Scheduled retraining: weekly (Monday 02:00 UTC)
- Manual retraining: workflow dispatch
- Push-triggered validation: regular quality checks and tests

## Artifacts and Lineage

Each successful run publishes:

- `models/imx8-compatible-sepsis.tflite`
- `pipeline_report.md`
- `model_metadata.json`
- `train_and_convert.log`

Lineage fields captured in reports/metadata include:

- Commit SHA
- Synthea version
- Training parameters (population, seed, epochs, batch size)
- Model size and quantization details
- Validation metrics

## Versioning Strategy

- Canonical runtime path: `models/imx8-compatible-sepsis.tflite`
- CI auto-commit: model changes are committed to `main`
- CI model tag format: `models/<synthea-version>-<timestamp>`

Operationally, Yocto should pin to a commit SHA that corresponds to a verified
model tag for deterministic builds.

## Failure and Quality Gates

The pipeline is considered failed when any of the following occur:

- Synthea generation or flattening failure
- Training/quantization failure
- Validation/test failure
- Artifact publication failure

Recommended local gate before pushing changes:

```bash
tools/check_ci.sh
```

## Related Documents

- Yocto consumption and recipe example: `docs/YOCTO_INTEGRATION.md`
- Contract lifecycle and compatibility policy: `docs/contract-pinning.md`
- Runtime model card: `models/README.md`

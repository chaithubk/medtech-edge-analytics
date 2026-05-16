# Pipeline Internals: Training to Deployable Edge Model

This document is the source of truth for how the CI pipeline produces and versions
the model consumed by edge and Yocto builds.

## Scope

- Training data source: synthetic FHIR records from Synthea with sepsis module strictly enforced (SYNTHEA_MODULES="sepsis", SYNTHEA_PATIENTS=2500) for reliable positive/negative class balance
- Processing: FHIR flattening into tabular ML dataset with robust multi-stage imputation (median/mean/fallback) for all vital features
- Output artifact: `models/imx8-compatible-sepsis.tflite`
- Delivery: model committed to git and version-tagged by CI

## End-to-End Flow

1. Generate synthetic FHIR data with Synthea (sepsis module only) into `data/raw_fhir/fhir/`.
2. Flatten FHIR records into `data/processed/dataset.csv` via `src/flatten_fhir.py` (robust imputation).
3. Train and validate model in `src/train_and_convert.py` (strict class-balance gate: aborts if only one class present).
4. Quantize/export TensorFlow Lite model to `models/imx8-compatible-sepsis.tflite` (int8, NPU-compatible).
5. Publish reports and metadata (`pipeline_report.md`, `model_metadata.json`, logs).
6. If model changed, CI commits updated model and creates model version tag.

## CI Triggering and Control

- Manual retraining: workflow dispatch (manual trigger only).
- Scheduled retraining was removed from the automated pipeline to avoid
	unreviewed model updates; retraining runs are intended to be run manually
	when an operator wants to refresh the model.
- When a trained model changes, CI now creates or updates a stable pull request
	(`model-update-sepsis-model`) with the updated artifact for human review and
	merge. The pipeline will continue to create a model tag in the format
	`models/<synthea-version>-<timestamp>` for release lineage.
	(Note: CI no longer pushes changes directly to `main` — pull requests are
	used to respect branch protection rules.)

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
- CI model tag format: `models/<synthea-version>-<timestamp>`

Operationally, Yocto should pin to a commit SHA that corresponds to a verified
model tag for deterministic builds. Note that model updates are now surfaced
through a reviewable pull request (`model-update-sepsis-model`) rather than an
automatic push to `main`.

## Failure and Quality Gates


The pipeline is considered failed when any of the following occur:

- Synthea generation or flattening failure
- Training/quantization failure
- Validation/test failure
- Artifact publication failure
- Training data contains only a single class (all healthy or all sepsis): pipeline aborts with clear error


Recommended local gate before pushing changes:

```bash
tools/check_ci.sh
```

## Related Documents

- Yocto consumption and recipe example: `docs/YOCTO_INTEGRATION.md`
- Contract lifecycle and compatibility policy: `docs/contract-pinning.md`
- Runtime model card: `models/README.md`

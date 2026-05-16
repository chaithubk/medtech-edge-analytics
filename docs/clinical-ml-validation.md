# Clinical ML Validation for Sepsis Detection (Synthea Pipeline)

- All data used for model development is generated using Synthea, simulating the “sepsis” clinical module.
- The pipeline flattens FHIR data to extract time-series features (vitals, labs) robustly, handling missingness and Synthea output variations.
- Model training, validation, and quantization are fully automated and reproducible via CI.
- Validation includes:
  - Data integrity checks (preflight FHIR inspection)
  - Regression/unit tests for flattening and feature extraction
  - End-to-end test runs with synthetic data
- All artifacts (model, dataset, logs, reports) are versioned and uploaded for traceability.

## How Validation is Performed

- The pipeline includes preflight checks to ensure FHIR files are present and valid before training.
- Regression and unit tests are run on the flattening script to ensure correct feature extraction.
- Model validation is performed using a hold-out validation set from the synthetic data.
- Reports and logs are generated for every run, documenting metrics and any issues.

See `pipeline-internals.md` for a full technical walkthrough.

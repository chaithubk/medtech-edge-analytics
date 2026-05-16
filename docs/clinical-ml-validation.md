# Clinical ML Validation for Sepsis Detection

All model development uses Synthea-generated synthetic data, with the sepsis module strictly enforced to guarantee positive/negative class balance. The pipeline includes automated validation: preflight FHIR checks, robust imputation for all vital features, a strict class-balance gate before training, regression/unit tests for feature extraction, and end-to-end tests. All artifacts are versioned for traceability. See `docs/pipeline-internals.md` for details.

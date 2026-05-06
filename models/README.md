# Sepsis Risk Model Card

## Model Summary

- Model name: Sepsis Risk Predictor
- Task: Binary risk scoring for sepsis likelihood
- Runtime: TensorFlow Lite
- Input shape: `(1, 20)`
- Output shape: `(1, 1)`
- Output semantics: probability-like score in the range `[0, 1]`

## Artifacts

- `sepsis_model.tflite`: default runtime artifact used by the service
- `sepsis_model_qemu.tflite`: QEMU/embedded compatibility artifact

If target runtime constraints differ (for example older operator support), regenerate `sepsis_model_qemu.tflite` with `tools/convert_model_for_qemu.py` from the source model.

## Intended Use

This model is intended for edge analytics experimentation and integration testing in sepsis-risk workflows.

- Intended users: engineering and data teams validating device-side inference
- Intended environment: on-device inference service connected to MQTT telemetry
- Out of scope: autonomous diagnosis, treatment recommendation, or direct clinical decision authority

## Inputs

The 20 engineered features are derived from the rolling vital buffer:

- Heart rate statistics: mean, std, min, max, trend
- Systolic blood pressure statistics: mean, std, min, max, trend
- Diastolic blood pressure statistics: mean, std, min, max, trend
- Oxygen saturation mean
- Respiratory rate mean and trend
- Lactate mean
- Composite SIRS and qSOFA mean score

Note: Temperature and additional oxygen trend metrics are available in service statistics but excluded from the model vector to preserve the expected input contract.

## Outputs and Thresholding

Inference output is mapped to user-facing risk levels:

- `LOW`: `< 0.30`
- `MODERATE`: `0.30` to `0.70`
- `HIGH`: `> 0.70`

The service additionally reports:

- `risk_score` in percent (`0` to `100`)
- `confidence` (raw model output)
- `model_latency_ms`

## Data and Training Status

This repository ships inference artifacts only. Training code and lineage metadata are not included here.

- Current status: development artifact for integration and performance testing
- Validation status: requires formal clinical and external cohort validation before clinical deployment

## Performance Characteristics

Operational target in this project:

- Inference latency target: `< 100 ms` in typical edge environments

Published clinical metrics (AUROC, sensitivity, specificity, calibration) should be added only once evaluated on governed datasets.

## Risks and Limitations

- No documented training-data provenance in this repository
- No calibration report included in current artifact package
- Single-model approach without ensemble redundancy
- Prediction quality depends on telemetry quality, cadence, and schema compliance

## Governance Recommendations

- Version and register each model artifact with immutable metadata
- Track training dataset versions and label quality metrics
- Add post-deployment drift and calibration monitoring
- Define threshold management policy with clinical stakeholders

## Change Log

- Current: model card aligned with active inference pipeline and payload contract
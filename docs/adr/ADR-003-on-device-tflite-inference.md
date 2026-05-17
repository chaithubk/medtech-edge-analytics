# ADR-003 — On-Device TFLite Inference for Sepsis Risk Detection

| Field | Value |
|---|---|
| **ADR ID** | ADR-003 |
| **Title** | On-Device TFLite Inference for Sepsis Risk Detection |
| **Status** | Accepted |
| **Date** | 2026-05-13 |
| **Deciders** | MedTech R&D, Systems Architect, Clinical Informatics Lead |
| **Affected Repos** | `medtech-edge-analytics`, `medtech-device-os` |

---

## Context

The MedTech sepsis detection system must continuously evaluate incoming vitals and produce a risk score. The fundamental architectural question is: **where should inference execute?**

Three options were considered:

### Option A: Cloud Inference

Vitals are forwarded to a cloud-hosted ML inference endpoint (e.g., AWS SageMaker, Azure ML, or a custom FastAPI inference service). The risk score is returned to the device and displayed on the clinician dashboard.

### Option B: On-Device Full TensorFlow Inference

The full TensorFlow (non-Lite) runtime is deployed on the edge device. The model is loaded and inference runs locally without any cloud dependency.

### Option C: On-Device TensorFlow Lite (TFLite) Inference

A quantized TFLite model is deployed on the edge device. Inference runs locally using the TFLite interpreter, which is optimized for ARM architecture and runs within a minimal memory footprint.

---

## Decision

**On-device TensorFlow Lite inference (Option C) is selected.**

The sepsis risk model is compiled to `.tflite` format, packaged as `/app/models/sepsis_model.tflite`, and invoked by the `medtech-edge-analytics` service using the TFLite Python interpreter. Inference never leaves the device boundary. Results are published locally over MQTT. The cloud backend receives prediction events asynchronously, but the alarm path is entirely local.

---

## Rationale

### 1. HIPAA Privacy-by-Design

Patient vitals are Protected Health Information (PHI). Transmitting vitals to a cloud inference endpoint over a potentially shared hospital network — even encrypted — creates a data-at-rest risk on cloud infrastructure, a de-identification failure risk if cloud logging captures payloads, and a HIPAA BAA obligation with every cloud provider in the inference path.

On-device inference eliminates all three risks architecturally. No vitals leave the device boundary. Privacy is not a policy control or a contractual guarantee — it is a physical constraint.

> **Note:** The current platform uses entirely synthetic data. In a production deployment, this architectural decision becomes a patient-safety and regulatory compliance control, not merely a design preference.

### 2. Deterministic Sub-100ms Latency

IEC 60601-1-8 §6.3 requires that physiological alarm conditions be presented to the alarm system within a bounded time from the triggering condition. Cloud inference introduces three latency variables that cannot be deterministically bounded: network round-trip time, cloud endpoint queue depth, and cold-start latency.

TFLite inference on ARM64 (NXP i.MX8MP class hardware) completes within 10–40 ms under validated test conditions. End-to-end (vitals received → risk score published) is consistently < 100 ms. This bound is enforceable and testable in CI using QEMU ARM64 emulation.

### 3. Offline Resilience

Clinical networks are unreliable. Planned maintenance windows, VLAN misconfigurations, and network hardware failures are routine. A cloud-inference architecture silences the sepsis alarm system during every network outage. On-device inference means the alarm system continues to function in complete network isolation — even air-gapped environments.

This directly addresses the ISO 14971 hazard: **"cloud network unavailability causes missed sepsis alarm."** On-device inference eliminates this hazard class entirely rather than mitigating it.

### 4. TFLite vs. Full TensorFlow

Full TensorFlow cannot run within the memory budget of a constrained ARM64 device (target: < 256 MB RSS for the full `medtech-edge-analytics` service). TFLite's quantized INT8 model format reduces the sepsis model size by approximately 4× versus FP32, enabling it to run within Yocto image constraints without a dedicated ML accelerator. TFLite also provides ARM-optimized XNNPACK delegate support for NEON SIMD acceleration on i.MX8MP.

### 5. Model Swappability

The `MODEL_PATH` environment variable enables model updates without code changes. A new `.tflite` artifact can be deployed via OTA update to the Yocto image, passing the same CI validation gates, without modifying the inference service binary. This is aligned with the FDA's Predetermined Change Control Plan framework for AI/ML-based SaMD.

---

## Consequences

### Positive

- Zero PHI leaves the device during inference (privacy-by-design, not policy-dependent)
- Deterministic < 100 ms inference latency, bounded and testable in CI
- Alarm system operates fully offline — zero dependency on cloud availability
- Model updates via `MODEL_PATH` without code change
- ARM-optimized TFLite XNNPACK delegate provides NEON SIMD acceleration on production hardware
- Small footprint enables deployment within Yocto image memory budget

### Negative

- Model accuracy is bounded by the training data and the quantization loss from FP32 → INT8 conversion (~1–2% accuracy reduction, acceptable for screening use case)
- Model retraining and deployment requires a formal OTA workflow (planned for v3.x)
- TFLite does not support all TensorFlow ops natively; model architecture is constrained to ops in the TFLite op subset

### Neutral

- The cloud backend (`medtech-telemetry-cloud`) receives prediction events asynchronously for population analytics; it is not in the alarm path

---

## Alternatives Considered

| Alternative | Reason Rejected |
|---|---|
| Cloud inference (AWS SageMaker / custom endpoint) | Network-latency non-determinism incompatible with IEC 60601-1-8 alarm timing; PHI transmission risk; offline failure mode unacceptable |
| Full TensorFlow on-device | Memory footprint exceeds Yocto image budget on constrained ARM64; startup time excessive |
| ONNX Runtime on-device | Viable alternative, but TFLite provides better ARM64 optimization and broader MedTech edge deployment precedent |
| Rule-based threshold alerting (no ML) | Insufficient sensitivity for early sepsis onset; misses multi-variate patterns not captured by simple thresholds; not aligned with SOFA/NEWS2 scoring clinical evidence base |

---

## Model Governance Considerations

| Governance Area | Current State | Production Requirement |
|---|---|---|
| Model training data | Synthetic (Synthea-modeled) | Retrospective clinical data with IRB approval |
| Model validation | CI regression tests on labeled fixtures | Clinical validation study (sensitivity/specificity on holdout set) |
| Model version tracking | Logged at startup; `model_latency_ms` in every payload | `model_version` field in every prediction payload; fleet version audit |
| Model update mechanism | Manual `MODEL_PATH` update + image rebuild | OTA signed model artifact delivery |
| Regulatory classification | Not submitted | FDA 510(k) or De Novo for AI/ML-based SaMD (Class II) |

---

## Standards References

| Standard | Relationship to This Decision |
|---|---|
| **IEC 60601-1-8:2006+AMD1:2012 §6.3** | Alarm response time requirement drives the < 100 ms inference latency target. On-device inference is the only architecture that satisfies this deterministically. |
| **HIPAA §164.312 (Technical Safeguards)** | On-device inference eliminates the transmission security and access control obligations associated with cloud PHI processing. |
| **ISO 14971:2019 §7** | On-device inference eliminates the hazard "cloud unavailability causes missed sepsis alarm." This is a hazard elimination, not a risk mitigation — the highest-priority risk control. |
| **FDA AI/ML-Based SaMD Action Plan (2021)** | `model_latency_ms` in every prediction payload provides the monitoring data stream required by the Predetermined Change Control Plan. Model version logging at startup supports the audit trail. |
| **IEC 62304:2015 §5.5** | `MODEL_PATH` configurability constitutes a software configurable item (SCI) subject to change control. TFLite runtime version is a software item under configuration management. |

---

## Review Date

This decision should be revisited if:
- NXP i.MX8MP NPU (Neural Processing Unit) drivers become stable in Yocto — enabling hardware-accelerated inference that may warrant a runtime switch from TFLite CPU to NPU delegate
- The model complexity grows beyond the TFLite op subset (e.g., transformer architectures), requiring ONNX Runtime or full TF
- A federated learning requirement is introduced, necessitating a cloud-coordinated inference architecture

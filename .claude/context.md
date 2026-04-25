# medtech-edge-analytics: Context for AI Models

## Project Overview

**Name**: MedTech Edge Analytics  
**Repository**: chaithubk/medtech-edge-analytics  
**Owner**: krishna chaithanya balakavi  
**Stage**: 1 (MVP - Sepsis Detection)  
**Tech Stack**: Python 3.11, TensorFlow Lite, Paho MQTT, NumPy  

## Business Context

**Problem**: Clinical alarm fatigue from false sepsis alerts  
**Solution**: Edge AI preprocessing reduces false positives  
**Value**: Clinician time saved, better patient outcomes  
**Risk**: Model accuracy < 90%, latency >100ms  

## Technical Goals (Stage 1)

1. ✅ Load & run pre-trained TFLite sepsis model
2. ✅ Maintain 360-point vital history (1 hour rolling window)
3. ✅ Extract sepsis risk features (20+ engineered features)
4. ✅ Subscribe to vitals via MQTT, publish predictions
5. ✅ Achieve <100ms inference latency
6. ✅ >80% test coverage, 0 memory leaks
7. ✅ Run entirely on-device (no cloud)

## Integration Points

**Upstream (Data Source)**:
- medtech-vitals-publisher → MQTT topic: `medtech/vitals/latest`
- Input: JSON with HR, BP, O2, Temp, Quality, Timestamp

**Downstream (Alert Consumer)**:
- medtech-clinician-ui ← MQTT topic: `medtech/predictions/sepsis`
- Output: JSON with Risk Score (0-100), Confidence, Feature Contributions

## Success Metrics

| Metric | Target | Validation |
|--------|--------|-----------|
| Model Inference Time | <100ms | Valgrind profiling |
| Memory Usage | <100MB | psutil monitoring |
| Model Accuracy | >90% | Test dataset validation |
| False Positive Rate | <10% | Clinical review |
| Test Coverage | >80% | pytest coverage |
| Uptime | >99.9% | Auto-reconnect logic |

## Design Principles

1. **Edge-First**: All processing on-device, minimal latency
2. **Deterministic**: Reproducible results, no randomness in inference
3. **Lightweight**: <5MB model, low CPU usage
4. **Interpretable**: Features explainable to clinicians
5. **Robust**: Graceful degradation if model fails
6. **Testable**: Mock TFLite for unit tests (no real model needed)

## Key Files

- `src/inference/tflite_model.py` - Model wrapper
- `src/inference/vital_buffer.py` - 360-point buffer
- `src/inference/sepsis_scorer.py` - Feature engineering + scoring
- `src/mqtt/mqtt_client.py` - MQTT integration
- `tests/test_*.py` - Unit tests
- `models/sepsis_model.tflite` - Pre-trained model (placeholder)

## Assumptions

- Vitals arrive every 10 seconds (consistent cadence)
- Model is pre-trained and provided (no training in Stage 1)
- MQTT broker is available at localhost:1883
- Python 3.11+ with NumPy, TensorFlow Lite Runtime
- <100ms latency is acceptable for clinical workflows

## Constraints

- ❌ No cloud dependency (must work offline)
- ❌ No threads/async (single-threaded, deterministic)
- ❌ No large external libraries (stay lightweight)
- ❌ No hardcoded paths (use environment variables)
- ❌ No global state (pure functions where possible)

## Next Steps (After Stage 1)

1. Collect real patient data
2. Label cohorts (sepsis vs. non-sepsis)
3. Retrain model on real data
4. Add explainability (SHAP)
5. Deploy to production devices (i.MX8MP)
6. Clinical validation study
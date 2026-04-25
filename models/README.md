# Sepsis Detection Model

## Model Card

**Model Name**: Sepsis Risk Predictor v0.1  
**Framework**: TensorFlow Lite  
**Format**: Quantized INT8  
**Size**: <5MB  
**Latency**: <100ms (ARM CPU)  

## Input

- **Shape**: (1, 20)
- **Type**: float32
- **Features**: 20 engineered features from vital history
  - Heart rate trend
  - Blood pressure variability
  - O2 saturation trend
  - Temperature dynamics
  - Composite risk indicators

## Output

- **Shape**: (1, 1)
- **Type**: float32
- **Range**: 0.0 - 1.0
- **Interpretation**: Sepsis probability
  - 0.0 - 0.3: Low risk
  - 0.3 - 0.7: Moderate risk
  - 0.7 - 1.0: High risk

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | >90% |
| Sensitivity | >85% |
| Specificity | >90% |
| False Positive Rate | <10% |
| Inference Time | <100ms |

## Assumptions

- Vitals arrive every 10 seconds
- Model input is pre-processed features
- Single prediction per inference
- Stateless (no sequence dependence)

## Limitations

- Trained on synthetic data (Stage 1)
- Requires real-world validation (Stage 2)
- Single model (ensemble recommended for production)
- No temporal context (uses current window only)

## Future Work (Stage 2+)

- Retrain on real patient cohorts
- Add explainability (SHAP)
- Ensemble with other models
- Temporal modeling (LSTM)
- Continuous learning from feedback
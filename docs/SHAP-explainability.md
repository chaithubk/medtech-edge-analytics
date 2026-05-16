# SHAP Explainability for Sepsis Model

- SHAP (SHapley Additive exPlanations) is used to interpret model predictions.
- The pipeline supports SHAP analysis on synthetic data, providing feature importance and model transparency.
- SHAP results are included in the model reports for clinical interpretability.

## How SHAP Explainability is Achieved

- SHAP values are computed for the trained model using synthetic test data.
- The pipeline includes scripts/notebooks to generate SHAP plots and feature importance rankings.
- SHAP results are included in the model reports for transparency.

See `pipeline-internals.md` for a full technical walkthrough.

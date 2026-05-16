# Case Study: Sepsis Prediction with Synthetic Data

- Synthetic patient data is generated using Synthea’s “sepsis” module.
- FHIR data is flattened to a tabular format, extracting relevant features for sepsis detection.
- A neural network is trained and quantized for edge deployment.
- The workflow ensures full reproducibility, from data generation to model export.
- All steps are documented and artifacts are available for review.

## How the Case Study Pipeline Works

- Synthea is run with the "sepsis" module to generate synthetic patient data.
- FHIR files are flattened to extract features using a custom script.
- The dataset is used to train and validate a neural network for sepsis prediction.
- The trained model is quantized and exported for edge deployment.
- All steps are automated and reproducible via the workflow.

See `pipeline-internals.md` for a full technical walkthrough.

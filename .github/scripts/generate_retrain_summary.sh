#!/usr/bin/env bash
set -e

mkdir -p artifacts/reports
SYNTHEA_VERSION=$(cat synthea_version.txt)
RETRAIN_TRIGGER="${GITHUB_EVENT_NAME}"
MODEL_SIZE=$(stat -f%z models/imx8-compatible-sepsis.tflite 2>/dev/null || stat -c%s models/imx8-compatible-sepsis.tflite 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > artifacts/reports/retrain_summary.md << EOF
# Automated Retraining Summary

- **Timestamp:** $TIMESTAMP
- **Synthea Version:** $SYNTHEA_VERSION
- **Retraining Trigger:** $RETRAIN_TRIGGER
- **Model Size:** $MODEL_SIZE bytes
- **Git SHA:** ${GITHUB_SHA}
- **Workflow Run:** [View Run](${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID})

## Training Configuration
- Synthea Patients: ${SYNTHEA_PATIENTS}
- Synthea Seed: ${SYNTHEA_SEED}
- Run ID: ${GITHUB_RUN_ID}

## Artifacts Generated
- `models/imx8-compatible-sepsis.tflite` (quantized model)
- `artifacts/reports/pipeline_report.md` (detailed training report)
- `artifacts/reports/model_metadata.json` (model metadata)
- `artifacts/reports/train_and_convert.log` (training logs)

## Next Steps
- Review model performance metrics in `model_metadata.json`
- Compare accuracy with previous versions
- Deploy to edge devices if accuracy is acceptable
EOF

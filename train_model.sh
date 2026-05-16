#!/bin/bash
# Train sepsis detection model with automatic dataset selection
# 
# Automatically handles:
# 1. PhysioNet Challenge 2019 data (if available)
# 2. Synthea FHIR data (if PhysioNet not found)
# 3. Missing data (with helpful error messages)

set -e

echo "=== Sepsis Detection Model Training Pipeline ==="
echo


# Check what data is available
PHYSIONET_DIR="data/raw_physionet"
FHIR_DIR="data/raw_fhir/fhir"
PROCESSED_DIR="data/processed"

PHYSIONET_COUNT=$(ls "$PHYSIONET_DIR"/*.csv 2>/dev/null | wc -l || echo 0)
FHIR_COUNT=$(ls "$FHIR_DIR"/*.json 2>/dev/null | wc -l || echo 0)

echo "Data Availability:"
echo "  PhysioNet files: $PHYSIONET_COUNT"
echo "  FHIR bundles: $FHIR_COUNT"
echo

# Process available data
if [ "$PHYSIONET_COUNT" -gt 0 ]; then
    echo "→ PhysioNet data detected. Loading..."
    python3 src/load_physionet_data.py
    echo "✓ PhysioNet data processed"
    echo
elif [ "$FHIR_COUNT" -gt 0 ]; then
    echo "→ FHIR data detected. Flattening..."
    python3 src/flatten_fhir.py
    echo "✓ FHIR data processed"
    echo
    python3 scripts/process_fallback.py
else
    echo "✗ ERROR: No dataset found!"
    echo
    echo "Attempting to download PhysioNet sample for CI/dev..."
    python3 scripts/process_fallback.py
fi

# Train model
echo "=== Training Model ==="
echo
python3 src/train_and_convert.py
echo
echo "✓ Training complete!"
echo
echo "Output files:"
echo "  Model: models/imx8-compatible-sepsis.tflite"
echo "  Report: artifacts/reports/pipeline_report.md"
echo "  Metadata: artifacts/reports/model_metadata.json"
echo
echo "View report: cat artifacts/reports/pipeline_report.md"

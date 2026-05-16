#!/bin/bash
# Train sepsis detection model with automatic fallback to sample data
#
# Flow:
#   1. Generate Synthea FHIR data (10,000 patients)
#   2. Flatten FHIR to CSV
#   3. If zero sepsis cases: merge with committed sample data
#   4. Train int8 model for i.MX 8 NPU edge device
#

set -e

echo "=== Sepsis Edge Device Model Training Pipeline ==="
echo

# Configuration
SYNTHEA_VERSION="3.3.0"
SYNTHEA_PATIENTS="10000"
SYNTHEA_SEED="12345"
SYNTHEA_MODULES="sepsis"
FHIR_DIR="data/raw_fhir"
PROCESSED_CSV="data/processed/dataset.csv"

# Step 1: Generate Synthea FHIR data
echo "Step 1: Generating Synthea FHIR cohort ($SYNTHEA_PATIENTS patients)..."
mkdir -p "$FHIR_DIR"

if ! command -v java &> /dev/null; then
    echo "✗ ERROR: Java not found. Required to run Synthea."
    exit 1
fi

cd /tmp
SYNTHEA_JAR="synthea-with-dependencies.jar"
if [ ! -f "$SYNTHEA_JAR" ]; then
    echo "  Downloading Synthea $SYNTHEA_VERSION..."
    curl -fsSL -o "$SYNTHEA_JAR" \
        "https://github.com/synthetichealth/synthea/releases/download/v${SYNTHEA_VERSION}/synthea-with-dependencies.jar"
fi

java -jar "$SYNTHEA_JAR" \
    -p "$SYNTHEA_PATIENTS" \
    -s "$SYNTHEA_SEED" \
    -m "$SYNTHEA_MODULES" \
    --exporter.baseDirectory="$PWD/$FHIR_DIR" 2>&1 | tail -20

cd - > /dev/null
echo "✓ Synthea generation complete"
echo

# Step 2: Flatten FHIR to CSV
echo "Step 2: Flattening FHIR data to CSV..."
python src/flatten_fhir.py
echo "✓ FHIR flattening complete"
echo

# Step 3: Process sample data (includes quality clinical cases)
echo "Step 3: Applying sample data for model training..."
python src/process_fallback.py
echo "✓ Dataset preparation complete"
echo

# Step 4: Train and export model
echo "Step 4: Training int8 model for i.MX 8 NPU..."
python src/train_and_convert.py

echo
echo "=== Training Pipeline Complete ==="
echo "Model exported to: models/imx8-compatible-sepsis.tflite"
ls -lh models/imx8-compatible-sepsis.tflite
echo
echo "✓ Training complete!"
echo
echo "Output files:"
echo "  Model: models/imx8-compatible-sepsis.tflite"
echo "  Report: artifacts/reports/pipeline_report.md"
echo "  Metadata: artifacts/reports/model_metadata.json"
echo
echo "View report: cat artifacts/reports/pipeline_report.md"

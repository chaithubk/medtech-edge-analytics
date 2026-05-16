#!/bin/bash
# Download a small sample of PhysioNet 2019 Sepsis data for CI/dev use
# Downloads only the first 10 patient files from Set A (open access)

set -e

PHYSIONET_URL="https://physionet.org/static/published-projects/challenge-2019/physionet-challenge-2019-training-set-a.zip"
PHYSIONET_DIR="data/raw_physionet"

mkdir -p "$PHYSIONET_DIR"
cd "$PHYSIONET_DIR"

if [ ! -f physionet-challenge-2019-training-set-a.zip ]; then
    echo "Downloading PhysioNet Challenge 2019 Set A..."
    wget -q "$PHYSIONET_URL"
fi

if [ ! -d training_set_a ]; then
    echo "Unzipping PhysioNet data..."
    unzip -q physionet-challenge-2019-training-set-a.zip
fi

# Copy only the first 10 CSVs for quick CI/dev runs
ls training_set_a/*.psv | head -n 10 | while read f; do
    # Convert PSV to CSV for loader compatibility
    csv_name="$(basename "$f" .psv).csv"
    awk -F'|' 'BEGIN{OFS=","} NR==1{gsub(/\r/,"",$0)} {print $0}' "$f" > "$csv_name"
done

echo "Sample PhysioNet data ready in $PHYSIONET_DIR"

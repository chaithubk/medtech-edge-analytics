import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf

model_path = Path("models/imx8-compatible-sepsis.tflite")
train_script = Path("src/train_and_convert.py")
report_dir = Path("artifacts/reports")
train_log_path = report_dir / "train_and_convert.log"
report_dir.mkdir(parents=True, exist_ok=True)

constants = {}
source = train_script.read_text(encoding="utf-8")
tree = ast.parse(source)
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "EPOCHS",
                "BATCH_SIZE",
                "RANDOM_SEED",
                "REPRESENTATIVE_DATASET_SIZE",
            }:
                constants[target.id] = ast.literal_eval(node.value)

interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Defensive check to ensure input_details/output_details are dicts (for pylint)
if not isinstance(input_details, dict):
    raise TypeError(f"input_details is not a dict: {type(input_details)}")
if not isinstance(output_details, dict):
    raise TypeError(f"output_details is not a dict: {type(output_details)}")

model_bytes = model_path.read_bytes()
size_bytes = len(model_bytes)
# Flexible regex to extract all metrics from TensorFlow epoch lines
metric_regex = re.compile(r"([a-zA-Z0-9_]+): ([0-9.eE+-]+)")
parsed_metrics = {}
if train_log_path.exists():
    log_text = train_log_path.read_text(encoding="utf-8", errors="ignore")
    # Find all lines with metrics (those with - metric: value - ...)
    for line in log_text.splitlines():
        # Only consider lines with at least one metric (skip epoch headers, etc.)
        if "-" in line and ":" in line:
            for metric, value in metric_regex.findall(line):
                try:
                    parsed_metrics[metric] = float(value)
                except Exception:
                    pass

metadata = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_path": str(model_path),
    "model_size_bytes": size_bytes,
    "model_size_kb": round(size_bytes / 1024.0, 2),
    "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
    "quantization": {
        "input_dtype": str(input_details["dtype"].__name__),
        "output_dtype": str(output_details["dtype"].__name__),
        "input_quantization": list(input_details.get("quantization", ())),
        "output_quantization": list(output_details.get("quantization", ())),
    },
    "training_config": {
        "epochs": constants.get("EPOCHS"),
        "batch_size": constants.get("BATCH_SIZE"),
        "random_seed": constants.get("RANDOM_SEED"),
        "representative_dataset_size": constants.get("REPRESENTATIVE_DATASET_SIZE"),
        "synthea_patients": "${SYNTHEA_PATIENTS}",
        "synthea_seed": "${SYNTHEA_SEED}",
    },
    "training_results": parsed_metrics,
}

metadata_path = report_dir / "model_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

report_path = report_dir / "pipeline_report.md"
# Ensure training_results is always a dict for mypy/pylint
training_results = metadata.get("training_results")
if not isinstance(training_results, dict):
    training_results = {}
training_results_md = "\n".join(f"- {k}: {v}" for k, v in sorted(training_results.items()))

# Defensive: ensure metadata is a dict for mypy
if not isinstance(metadata, dict):
    raise TypeError(f"metadata is not a dict: {type(metadata)}")
# Defensive: ensure all subfields are dicts for mypy
training_config = metadata.get("training_config")
if not isinstance(training_config, dict):
    training_config = {}
quantization = metadata.get("quantization")
if not isinstance(quantization, dict):
    quantization = {}

report = f"""# Synthea Sepsis Training Report

## Pipeline Outcome
- Status: success
- Generated model: `{metadata.get('model_path', '')}`
- Model size: {metadata.get('model_size_kb', '')} KB ({metadata.get('model_size_bytes', '')} bytes)
- SHA256: `{metadata.get('model_sha256', '')}`

## Training Configuration
- Synthea patients: {training_config.get('synthea_patients', '')}
- Synthea seed: {training_config.get('synthea_seed', '')}
- Epochs: {training_config.get('epochs', '')}
- Batch size: {training_config.get('batch_size', '')}
- Random seed: {training_config.get('random_seed', '')}
- Representative dataset size: {training_config.get('representative_dataset_size', '')}

## Training Results
{training_results_md if training_results_md else '- No metrics parsed from log'}

## Quantization Validation (TFLite)
- Input dtype: `{quantization.get('input_dtype', '')}`
- Output dtype: `{quantization.get('output_dtype', '')}`
- Input quantization: `{quantization.get('input_quantization', '')}`
- Output quantization: `{quantization.get('output_quantization', '')}`

## Evidence Artifacts
- `artifacts/reports/train_and_convert.log`
- `artifacts/reports/model_metadata.json`
- `models/imx8-compatible-sepsis.tflite`
"""
report_path.write_text(
    "\n".join(line.strip() for line in report.splitlines()) + "\n", encoding="utf-8"
)

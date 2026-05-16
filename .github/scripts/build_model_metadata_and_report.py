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

model_bytes = model_path.read_bytes()
size_bytes = len(model_bytes)
metric_patterns = {
    "loss": r"^\s*Loss:\s*([0-9]*\.?[0-9]+)",
    "compile_metrics": r"^\s*Compile_metrics:\s*([0-9]*\.?[0-9]+)",
}
parsed_metrics = {}
if train_log_path.exists():
    log_text = train_log_path.read_text(encoding="utf-8", errors="ignore")
    for key, pattern in metric_patterns.items():
        match = re.search(pattern, log_text, flags=re.MULTILINE)
        if match:
            parsed_metrics[key] = float(match.group(1))

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
report = f"""# Synthea Sepsis Training Report

## Pipeline Outcome
- Status: success
- Generated model: `{metadata['model_path']}`
- Model size: {metadata['model_size_kb']} KB ({metadata['model_size_bytes']} bytes)
- SHA256: `{metadata['model_sha256']}`

## Training Configuration
- Synthea patients: {metadata['training_config']['synthea_patients']}
- Synthea seed: {metadata['training_config']['synthea_seed']}
- Epochs: {metadata['training_config']['epochs']}
- Batch size: {metadata['training_config']['batch_size']}
- Random seed: {metadata['training_config']['random_seed']}
- Representative dataset size: {metadata['training_config']['representative_dataset_size']}

## Training Results
- Loss: {metadata['training_results'].get('loss', 'not parsed from log')}
- Compile metrics: {metadata['training_results'].get('compile_metrics', 'not parsed from log')}

## Quantization Validation (TFLite)
- Input dtype: `{metadata['quantization']['input_dtype']}`
- Output dtype: `{metadata['quantization']['output_dtype']}`
- Input quantization: `{metadata['quantization']['input_quantization']}`
- Output quantization: `{metadata['quantization']['output_quantization']}`

## Evidence Artifacts
- `artifacts/reports/train_and_convert.log`
- `artifacts/reports/model_metadata.json`
- `models/imx8-compatible-sepsis.tflite`
"""
report_path.write_text(
    "\n".join(line.strip() for line in report.splitlines()) + "\n", encoding="utf-8"
)

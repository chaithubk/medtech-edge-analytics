#!/usr/bin/env python3
"""Convert a TensorFlow model to a QEMU/ARM-friendly TFLite artifact.

Usage:
  python tools/convert_model_for_qemu.py \
    --input /path/to/saved_model_or_keras \
    --output models/sepsis_model_qemu.tflite \
    --mode float

Notes:
- This requires the ORIGINAL source model (SavedModel directory or Keras file).
- Converting from an existing .tflite file to another .tflite is not supported here.
- Use --mode float for broad compatibility.
- Use --mode int8 only when a representative dataset is available.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TensorFlow model for QEMU/ARM")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to source model: SavedModel directory or .keras/.h5 file.",
    )
    parser.add_argument(
        "--output",
        default="models/sepsis_model_qemu.tflite",
        help="Output .tflite path (default: models/sepsis_model_qemu.tflite).",
    )
    parser.add_argument(
        "--mode",
        choices=["float", "dynamic", "int8"],
        default="float",
        help="Conversion mode: float (max compatibility), dynamic, int8.",
    )
    return parser.parse_args()


def _load_converter(tf, input_path: Path):
    if input_path.is_dir():
        return tf.lite.TFLiteConverter.from_saved_model(str(input_path))

    suffix = input_path.suffix.lower()
    if suffix not in {".keras", ".h5"}:
        raise ValueError(
            "Unsupported input format. Provide a SavedModel directory, .keras, or .h5 file."
        )

    model = tf.keras.models.load_model(str(input_path))
    return tf.lite.TFLiteConverter.from_keras_model(model)


def _representative_dataset() -> Iterable[list]:
    # Replace with real calibration samples shaped (1, 20) float32.
    import numpy as np

    for _ in range(100):
        yield [np.zeros((1, 20), dtype=np.float32)]


def main() -> int:
    args = _parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model path not found: {input_path}")

    if input_path.suffix.lower() == ".tflite":
        raise ValueError(
            "Input is already .tflite. Re-conversion requires original SavedModel/Keras source."
        )

    import tensorflow as tf

    converter = _load_converter(tf, input_path)

    # Keep operator set strict for embedded targets.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_new_converter = True

    if args.mode == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif args.mode == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    print(f"Wrote: {output_path}")
    print(f"Size: {output_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

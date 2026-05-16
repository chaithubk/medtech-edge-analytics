tflite_model = converter.convert()
# Model Quantization for Edge Deployment

After training, the sepsis model is quantized to TensorFlow Lite (.tflite) format for fast, efficient edge inference. Quantization is performed in `src/train_and_convert.py` using a representative dataset to enable integer-only operations for embedded hardware.

The output artifact is always `models/imx8-compatible-sepsis.tflite`, validated for accuracy and compatibility. See `docs/pipeline-internals.md` for the full workflow and lineage.

# Edge Inference Benchmarking

- The quantized .tflite model is optimized for imx8/qemu hardware.
- Benchmarking includes:
  - Inference speed and resource usage on target hardware
  - Accuracy validation using synthetic test data
- Results are documented in pipeline reports and model metadata.

## How Edge Inference Benchmarking is Performed

- The quantized .tflite model is deployed to imx8/qemu hardware or emulator.
- Inference speed and accuracy are measured using synthetic test data.
- Resource usage (CPU, memory) is monitored during benchmarking.
- Results are compared to baseline and documented in reports.

See `pipeline-internals.md` for a full technical walkthrough.

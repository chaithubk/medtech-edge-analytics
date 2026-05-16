# Model Quantization for Edge Deployment

- After training, the sepsis model is quantized to TensorFlow Lite (.tflite) format.
- Quantization reduces model size and enables fast inference on edge devices (imx8/qemu).
- The quantized model is validated for accuracy and compatibility.
- The pipeline produces both the quantized model and detailed metadata (model parameters, quantization settings, performance metrics).

## How Model Quantization is Achieved

- After training, the Keras model is converted to TensorFlow Lite using the TFLite Converter in `src/train_and_convert.py`.
- Quantization is performed by providing a representative dataset (sample input data) to the converter, enabling integer-only operations for edge hardware.
- The quantized model is tested for accuracy to ensure minimal loss compared to the original model.
- The final `.tflite` file is compatible with imx8/qemu and can be deployed directly to edge devices.

**Example (Python):**
```python
import tensorflow as tf
# Load trained model
model = tf.keras.models.load_model('model.h5')
# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Provide representative dataset for quantization
converter.representative_dataset = ...
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

See `pipeline-internals.md` for a full step-by-step walkthrough.

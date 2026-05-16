#!/usr/bin/env python3
"""
Hardware-optimized ML engine for i.MX 8 edge deployment.

Trains a compact Keras model on scaled vital signs data and exports
as fully integer-quantized TFLite binary for i.MX 8 NPU delegation.
"""

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
DATASET_PATH = PROCESSED_DIR / "dataset.csv"
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"
TFLITE_OUTPUT_PATH = MODEL_DIR / "imx8-compatible-sepsis.tflite"

REPRESENTATIVE_DATASET_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 50
RANDOM_SEED = 42


def load_dataset() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load preprocessed dataset.

    Returns:
        Tuple of (full DataFrame, features array, labels array)
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    # Use all relevant numeric features from the schema
    feature_cols = [
        "hr",
        "bp_sys",
        "bp_dia",
        "o2_sat",
        "temperature",
        "respiratory_rate",
        "wbc",
        "lactate",
        "creatinine",
        "sirs_score",
        "qsofa_score",
    ]
    # Only keep columns that exist in the dataset
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].values.astype(np.float32)
    y = (
        df["sepsis"].values.astype(np.int32)
        if "sepsis" in df.columns
        else np.zeros(len(df), dtype=np.int32)
    )

    return df, X, y


def build_model(input_shape: int) -> keras.Model:
    """
    Build compact Sequential model optimized for edge deployment.

    Constraints:
    - Uses only TFLite-delegable operations (relu, logistic)
    - Minimal parameters for efficient i.MX 8 execution
    - No batch normalization or custom activations

    Args:
        input_shape: Number of input features

    Returns:
        Compiled Keras Sequential model
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_shape,), dtype=tf.float32),
            keras.layers.Dense(32, activation="relu", name="dense_1"),
            keras.layers.Dense(16, activation="relu", name="dense_2"),
            keras.layers.Dense(8, activation="relu", name="dense_3"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.AUC(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
        ],
    )

    return model


def representative_dataset_gen(
    X_representative: np.ndarray,
) -> Iterator[list[np.ndarray]]:
    """
    Generator for representative dataset during quantization calibration.

    This function feeds scaled training data into the TFLiteConverter
    for 8-bit integer calibration. Each yielded batch must be a tuple
    containing numpy arrays matching the model's input specification.

    Args:
        X_representative: Numpy array of representative samples (float32)

    Yields:
        Tuples containing single batches as float32 arrays
    """
    for i in range(0, len(X_representative), BATCH_SIZE):
        batch = X_representative[i : i + BATCH_SIZE].astype(np.float32)
        yield [batch]


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """
    Train model with class weight balancing for imbalanced sepsis data.

    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
    """
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

    print(f"Class weights: {class_weight_dict}")

    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        class_weight=class_weight_dict,
        verbose=1,
    )


def export_tflite_int8(
    model: keras.Model,
    X_representative: np.ndarray,
    output_path: Path,
) -> None:
    """
    Export model as fully integer-quantized TFLite binary for i.MX 8.

    Configuration:
    - TFLITE_BUILTIN_INT8: Forces 8-bit integer ops
    - inference_input_type: tf.int8 (explicit 8-bit integer inputs)
    - inference_output_type: tf.int8 (explicit 8-bit integer outputs)
    - representative_dataset: Calibration data for quantization ranges

    Args:
        model: Trained Keras model
        X_representative: Data for quantization calibration
        output_path: Path to save .tflite file
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(X_representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Exported fully integer-quantized model to {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")


def main():
    """Main training and export pipeline."""
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("Loading preprocessed dataset...")
    df, X, y = load_dataset()

    print(f"Dataset shape: {X.shape}")
    print(f"Feature columns: {list(df.columns)}")
    # Ensure y has only non-negative integers for bincount
    if np.any(y < 0):
        print(
            "Warning: Negative values found in target labels. "
            "Setting negatives to zero for bincount."
        )
        y_bincount = np.bincount(np.clip(y.astype(int), 0, None))
    else:
        y_bincount = np.bincount(y.astype(int))
    print(f"Target distribution: {y_bincount}")

    print("\nBuilding compact Keras model...")
    model = build_model(input_shape=X.shape[1])
    model.summary()

    print("\nTraining model with class-weight balancing...")
    train_model(model, X, y)

    print("\nEvaluating on full dataset...")
    results = model.evaluate(X, y, verbose=0)
    metric_names = ["loss"] + [m.name for m in model.metrics]
    for name, value in zip(metric_names, results):
        print(f"  {name.capitalize()}: {value:.4f}")

    print("\nSelecting representative calibration dataset...")
    representative_indices = np.random.choice(
        len(X),
        size=min(REPRESENTATIVE_DATASET_SIZE, len(X)),
        replace=False,
    )
    X_representative = X[representative_indices]
    print(f"Calibration samples: {len(X_representative)}")

    print("\nExporting fully integer-quantized TFLite model for i.MX 8...")
    export_tflite_int8(model, X_representative, TFLITE_OUTPUT_PATH)

    print("\nValidating TFLite interpreter...")
    interpreter = tf.lite.Interpreter(str(TFLITE_OUTPUT_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input dtype: {input_details[0]['dtype'].__name__}")
    print(f"  Output dtype: {output_details[0]['dtype'].__name__}")
    print(f"  Input quantization: {input_details[0].get('quantization', {})}")
    print(f"  Output quantization: {output_details[0].get('quantization', {})}")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()

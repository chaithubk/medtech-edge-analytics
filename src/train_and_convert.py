#!/usr/bin/env python3
"""
Hardware-optimized ML engine for i.MX 8 edge deployment.

Trains a compact Keras model on scaled vital signs data and exports
as fully integer-quantized TFLite binary for i.MX 8 NPU delegation.
"""

from pathlib import Path
from typing import Iterator, Tuple, cast

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray
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


def load_dataset() -> Tuple[pd.DataFrame, object, object]:  # type: ignore
    """
    Load preprocessed dataset.

    If dataset.csv doesn't exist, intelligently load from PhysioNet or FHIR.

    Returns:
        Tuple of (full DataFrame, features array, labels array)
    """
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}. Attempting to load from available sources...")
        try:
            from load_dataset import load_dataset as load_from_sources

            df = load_from_sources()[0]
            # Ensure a proper DataFrame type for static analysis
            df = pd.DataFrame(df)
            # Save for future runs
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATASET_PATH, index=False)
            print(f"✓ Loaded and saved dataset to {DATASET_PATH}")
        except Exception as e:
            raise FileNotFoundError(
                f"Dataset not found and could not load from available sources: {e}\n"
                "Please either:\n"
                "1. Download PhysioNet Challenge 2019:\n"
                "   https://physionet.org/content/challenge-2019/1.0.0/\n"
                "2. Generate Synthea data:\n"
                "   SYNTHEA_PATIENTS=5000 bash scripts/generate_synthea_data.sh\n"
                "See DATASET_SETUP.md for details."
            ) from e
    else:
        df = pd.read_csv(DATASET_PATH)
    # If dataset contains time-series per patient (patient_id + timestamp),
    # compute the same 20 engineered features used at runtime by
    # `src/inference/vital_buffer.py::VitalBuffer.get_all_features()`.
    # Otherwise fall back to per-row snapshot features (legacy behaviour).
    required_ts_cols = {"patient_id", "timestamp"}

    if required_ts_cols.issubset(set(df.columns)):
        # Ensure timestamp is numeric and sort per patient/time
        df = df.copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)
        df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

        # rolling window size (match VitalBuffer default window used at runtime)
        WINDOW_SIZE = 60

        def compute_stats(window_df: pd.DataFrame) -> dict:  # type: ignore
            def arr(col: str) -> NDArray[np.float64]:
                result = window_df[col].to_numpy(dtype=np.float64)
                return cast(NDArray[np.float64], result)  # type: ignore

            def trend(a: np.ndarray) -> float:
                if len(a) < 2:
                    return 0.0
                coeffs = np.polyfit(np.arange(len(a)), a, 1)
                return float(coeffs[0])

            hr = arr("hr") if "hr" in window_df else np.array([])
            bp_sys = arr("bp_sys") if "bp_sys" in window_df else np.array([])
            bp_dia = arr("bp_dia") if "bp_dia" in window_df else np.array([])
            o2 = arr("o2_sat") if "o2_sat" in window_df else np.array([])
            rr = arr("respiratory_rate") if "respiratory_rate" in window_df else np.array([])
            lactate = arr("lactate") if "lactate" in window_df else np.array([])
            sirs = arr("sirs_score") if "sirs_score" in window_df else np.array([])
            qsofa = arr("qsofa_score") if "qsofa_score" in window_df else np.array([])

            # helper safe reductions
            def mean(a: np.ndarray) -> float:
                return float(np.mean(a)) if a.size else 0.0

            def std(a: np.ndarray) -> float:
                return float(np.std(a)) if a.size else 0.0

            def amin(a: np.ndarray) -> float:
                return float(np.min(a)) if a.size else 0.0

            def amax(a: np.ndarray) -> float:
                return float(np.max(a)) if a.size else 0.0

            stats = {
                "hr_mean": mean(hr),
                "hr_std": std(hr),
                "hr_min": amin(hr),
                "hr_max": amax(hr),
                "hr_trend": trend(hr),
                "bp_sys_mean": mean(bp_sys),
                "bp_sys_std": std(bp_sys),
                "bp_sys_min": amin(bp_sys),
                "bp_sys_max": amax(bp_sys),
                "bp_sys_trend": trend(bp_sys),
                "bp_dia_mean": mean(bp_dia),
                "bp_dia_std": std(bp_dia),
                "bp_dia_min": amin(bp_dia),
                "bp_dia_max": amax(bp_dia),
                "bp_dia_trend": trend(bp_dia),
                "o2_mean": mean(o2),
                "rr_mean": mean(rr),
                "rr_trend": trend(rr),
                "lactate_mean": mean(lactate),
                "sirs_qsofa_mean": mean(sirs) + mean(qsofa),
            }
            return stats

        engineered_rows = []
        labels = []

        # group per patient and compute rolling-stat features for each sample
        for pid, group in df.groupby("patient_id"):
            # Use a rolling window ending at each index
            values = group.reset_index(drop=True)
            for idx in range(len(values)):
                start = max(0, idx - WINDOW_SIZE + 1)
                window_df = values.iloc[start : idx + 1]
                stats = compute_stats(window_df)
                # Build feature vector in the same order as VitalBuffer.get_all_features()
                vec = [
                    stats["hr_mean"],
                    stats["hr_std"],
                    stats["hr_min"],
                    stats["hr_max"],
                    stats["hr_trend"],
                    stats["bp_sys_mean"],
                    stats["bp_sys_std"],
                    stats["bp_sys_min"],
                    stats["bp_sys_max"],
                    stats["bp_sys_trend"],
                    stats["bp_dia_mean"],
                    stats["bp_dia_std"],
                    stats["bp_dia_min"],
                    stats["bp_dia_max"],
                    stats["bp_dia_trend"],
                    stats["o2_mean"],
                    stats["rr_mean"],
                    stats["rr_trend"],
                    stats["lactate_mean"],
                    stats["sirs_qsofa_mean"],
                ]
                engineered_rows.append(vec)
                labels.append(int(values.iloc[idx]["sepsis"]) if "sepsis" in values.columns else 0)

        X = np.array(engineered_rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        X = cast(np.ndarray, X)  # type: ignore
        y = cast(np.ndarray, y)  # type: ignore
        return df, X, y  # type: ignore
    else:
        # Legacy per-row snapshot features (keep backward compatible)
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
        feature_cols = [col for col in feature_cols if col in df.columns]

        from typing import cast

        X = df[feature_cols].values.astype(np.float32)
        y = (
            df["sepsis"].values.astype(np.int32)
            if "sepsis" in df.columns
            else np.zeros(len(df), dtype=np.int32)
        )
        X = cast(np.ndarray, X)
        y = cast(np.ndarray, y)
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

    if np.any(y < 0):
        print(
            "Warning: Negative values found in target labels. "
            "Setting negatives to zero for bincount."
        )
        y_bincount = np.bincount(np.clip(y.astype(int), 0, None))
    else:
        y_bincount = np.bincount(y.astype(int))

    print(f"Target distribution: {y_bincount}")

    unique_classes = np.unique(y.astype(int))
    if len(unique_classes) < 2:
        raise ValueError(
            f"FATAL: Training dataset contains only a single class "
            f"(classes={unique_classes.tolist()}). "
            "Cannot train a binary classifier without both positive (sepsis=1) and negative "
            "(sepsis=0) examples. "
            "Ensure Synthea generated data with SYNTHEA_MODULES='sepsis' and verify that FHIR "
            "bundles contain SNOMED code 91302003 for sepsis diagnoses. "
            f"Current class distribution: "
            f"{dict(zip(unique_classes.tolist(), map(int, y_bincount)))}"
        )

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

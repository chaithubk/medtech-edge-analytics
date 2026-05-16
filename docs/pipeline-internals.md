# Pipeline Internals: How the Sepsis Model is Built and Maintained

## 1. Synthea Dataset Generation
- **What:** Synthea is used to generate fully synthetic patient records for the "sepsis" module.
- **How:**
  - The pipeline downloads the Synthea JAR and runs it with parameters (module: sepsis, population size, seed, state).
  - Output is FHIR (JSON) files in `data/raw_fhir/fhir/`.
  - Example command:
    ```bash
    java -jar synthea-with-dependencies.jar -p 100 -s 42 -m sepsis --exporter.fhir.export=true Massachusetts
    ```

## 2. Data Flattening and Feature Extraction
- **What:** Converts FHIR files into a tabular dataset for ML.
- **How:**
  - `src/flatten_fhir.py` parses each FHIR file, extracting time-series features (vitals, labs).
  - Handles missing data and Synthea output variations robustly.
  - Output: `data/processed/dataset.csv`.

## 3. Model Training
- **What:** Trains a neural network to predict sepsis.
- **How:**
  - `src/train_and_convert.py` loads the CSV, splits into train/validation sets, and trains a Keras model.
  - Model architecture and parameters are set in the script.
  - Training logs and metrics are saved for review.

## 4. Model Quantization
- **What:** Converts the trained model to TensorFlow Lite (.tflite) for edge deployment.
- **How:**
  - The script quantizes the model, reducing size and optimizing for imx8/qemu.
  - Output: `models/imx8-compatible-sepsis.tflite`.

## 5. Reporting and Artifacts
- **What:** Documents the pipeline run and model details.
- **How:**
  - Generates `pipeline_report.md`, `model_metadata.json`, and logs.
  - All artifacts are uploaded for traceability.

## 6. Automated Retraining via Scheduled CI Runs

- **Scheduled Trigger:**  
  The pipeline is automatically triggered every Monday at 02:00 UTC to retrain the model with the latest Synthea data.
  ```yaml
  on:
    schedule:
      - cron: '0 2 * * 1'  # Weekly retraining: Every Monday at 02:00 UTC
  ```

- **Manual Trigger:**  
  You can also manually trigger retraining via GitHub Actions UI or `workflow_dispatch`.

- **Retraining Summary Report:**  
  Each retraining run generates a detailed summary (`retrain_summary.md`) that includes:
  - **Timestamp:** UTC time of the retraining run
  - **Synthea Version:** Version of Synthea used (e.g., 3.3.0)
  - **Retraining Trigger:** Why the retraining was initiated (scheduled, push, manual)
  - **Model Size:** Size of the quantized `.tflite` model
  - **Git SHA:** Commit hash for reproducibility
  - **Workflow Run:** Link to the GitHub Actions run for detailed logs
  - **Training Configuration:** Number of patients, seed, epochs, batch size, etc.
  - **Artifacts Generated:** List of all outputs (model, logs, reports)

- **Model Metadata and Accuracy Tracking:**  
  The pipeline captures and stores:
  - `model_metadata.json`: Detailed metadata including model size, quantization settings, training parameters, and loss/accuracy metrics.
  - `pipeline_report.md`: Human-readable report with training results and configuration.
  - `train_and_convert.log`: Raw training logs for debugging.

- **Artifact Storage:**  
  All artifacts are bundled and uploaded to GitHub Actions with a 30-day retention period. You can download and compare across retraining runs to track improvements.

- **Workflow Summary:**  
  After each run, a job summary is posted to GitHub Actions, showing:
  - Pipeline outcome and model path
  - Training configuration (Synthea patients, seed, epochs, batch size)
  - Training results (loss, metrics)
  - Quantization validation (input/output dtypes)
  - Retraining details (trigger, Synthea version, model size)

## 7. Data Sufficiency and Retraining Guidance

- **How much data is enough?**
  - For initial development, 100–1000 synthetic patients is typical. More data improves robustness.
  - Monitor model performance on validation/test sets. If accuracy is low or new clinical patterns emerge, generate more data.
  - Use the accuracy metrics in `model_metadata.json` to assess data sufficiency.

- **How often to retrain?**
  - **Automated:** Every Monday (or customize the cron schedule in the workflow).
  - **Manual:** Trigger manually if:
    - Synthea is updated (new modules, bugfixes)
    - Model performance degrades in production
    - New features or clinical modules are added
  - **Event-driven:** Monitor accuracy trends from retraining summaries; increase frequency if degradation is detected.

- **Comparing Retraining Runs:**
  - Download `model_metadata.json` and `retrain_summary.md` from two consecutive runs.
  - Compare:
    - Model size (should be similar across runs with same configuration)
    - Loss and accuracy metrics
    - Synthea version and training parameters
  - Use GitHub Actions artifact history to track trends over time.

## 8. Best Practices

- **Versioning:** Always use a fixed Synthea version (e.g., 3.3.0) for consistency. Update only when necessary and document the reason.
- **Reproducibility:** Document all parameters (population, seed, state, Synthea version) in the summary and workflow.
- **Monitoring:** Review retraining summaries regularly to identify performance trends or anomalies.
- **Validation:** Before deploying a new model, validate its predictions with independent test data.
- **Archival:** Keep retraining summaries and metadata for audit trails and model lineage.
- **Alerts:** Set up GitHub notifications for retraining failures to catch issues quickly.

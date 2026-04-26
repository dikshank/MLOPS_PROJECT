"""
retraining_dag.py
-----------------
Phase 4 | Executed: Local (inside Airflow Docker container)

Airflow DAG: melanoma_retraining_pipeline

Monitors for retraining triggers and orchestrates the retraining pipeline.

Trigger conditions (checked by check_trigger task):
    1. retrain_needed.flag file exists in logs/
       - written by FastAPI when misclassification rate > threshold
       - written by FastAPI when drift score > threshold
    2. Manual trigger from Airflow UI (always runs full pipeline)

Pipeline tasks:
    check_trigger
          ↓
    prepare_feedback_data   ← merge feedback images into training data
          ↓
    trigger_training        ← run train.py with updated data
          ↓
    evaluate_new_model      ← compare new model vs current Production
          ↓
    cleanup                 ← remove retrain flag, archive feedback data
"""

import os
import json
import shutil
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("retraining_dag")

# ── Paths (inside Docker container) ──────────────────────────────────────────
# These match the volumes mounted in docker-compose.yml
RETRAIN_FLAG_PATH   = Path("/opt/airflow/app_logs/retrain_needed.flag")
FEEDBACK_DATA_DIR   = Path("/opt/airflow/app_logs/feedback_data")
PROCESSED_DATA_DIR  = Path("/opt/airflow/data/processed/v1")
TRAINING_SCRIPT     = Path("/opt/airflow/training/src/train.py")

# All 3 model configs — retraining runs all models, champion/challenger picks best
TRAINING_CONFIGS = [
    Path("/opt/airflow/training/configs/config_v1_mobilenet.yaml"),
    Path("/opt/airflow/training/configs/config_v1_efficientnet.yaml"),
    Path("/opt/airflow/training/configs/config_v1_simplecnn.yaml"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Check trigger conditions
# ─────────────────────────────────────────────────────────────────────────────

def check_trigger(**context) -> str:
    """
    Check if retraining should proceed.

    Reads the retrain_needed.flag file written by FastAPI.
    If flag exists → proceed with retraining.
    If manually triggered → always proceed.

    Returns:
        str: Trigger reason ('misclassification', 'drift', or 'manual')

    Raises:
        ValueError: If no trigger condition is met (stops pipeline cleanly).
    """
    # Manual trigger always proceeds
    is_manual = context.get("dag_run") and \
                context["dag_run"].run_type == "manual"

    if RETRAIN_FLAG_PATH.exists():
        with open(RETRAIN_FLAG_PATH) as f:
            flag = json.load(f)
        reason = flag.get("reason", "unknown")
        logger.info(
            "✅ Retraining triggered | reason=%s | flag_time=%s",
            reason, flag.get("timestamp", "unknown")
        )
        return reason

    elif is_manual:
        logger.info("✅ Manual retraining triggered from Airflow UI")
        return "manual"

    else:
        logger.info(
            "ℹ️ No retraining trigger found. "
            "Flag file not present. Skipping pipeline."
        )
        raise ValueError(
            "No retraining trigger condition met. "
            "Pipeline stopped cleanly."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Prepare feedback data
# ─────────────────────────────────────────────────────────────────────────────

def prepare_feedback_data(**context) -> dict:
    """
    Merge confirmed feedback images into the training data manifests.

    Reads images from logs/feedback_data/malignant/ and logs/feedback_data/benign/
    and appends their relative paths to the train manifest CSV.

    Returns:
        dict: Summary of images added per class.
    """
    import pandas as pd

    manifest_path = PROCESSED_DATA_DIR / "manifests" / "train_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Train manifest not found: {manifest_path}. "
            f"Run data pipeline first."
        )

    df = pd.read_csv(manifest_path)
    new_rows = []

    for class_name in ["malignant", "benign"]:
        class_dir = FEEDBACK_DATA_DIR / class_name

        if not class_dir.exists():
            logger.info("No feedback images for class: %s", class_name)
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

        for img_path in images:
            # Copy to processed data folder
            dest_dir  = PROCESSED_DATA_DIR / "train" / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / img_path.name
            shutil.copy2(str(img_path), str(dest_path))

            # Add relative path to manifest
            relative_path = f"data/processed/v1/train/{class_name}/{img_path.name}"
            new_rows.append({
                "filepath": relative_path,
                "label":    class_name
            })

        logger.info(
            "Class [%s]: %d feedback images added to training data",
            class_name, len(images)
        )

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_df.to_csv(manifest_path, index=False)
        logger.info(
            "Train manifest updated: %d → %d rows",
            len(df), len(updated_df)
        )
    else:
        logger.warning("No feedback images found. Training on original data.")

    return {
        "original_rows":  len(df),
        "new_rows":       len(new_rows),
        "total_rows":     len(df) + len(new_rows)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Trigger training
# ─────────────────────────────────────────────────────────────────────────────

def trigger_training(**context) -> None:
    """
    Run the training script with the MobileNet config.

    This runs locally via subprocess. In demo mode, you can stop it
    after a few batches to show the pipeline fires correctly.
    """
    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found: {TRAINING_SCRIPT}"
        )

    for config in TRAINING_CONFIGS:
        if not config.exists():
            raise FileNotFoundError(
                f"Training config not found: {config}"
            )

        logger.info(
            "Starting retraining | script=%s | config=%s",
            TRAINING_SCRIPT, config
        )

        result = subprocess.run(
            ["python", str(TRAINING_SCRIPT), "--config", str(config)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max per model
        )

        if result.returncode != 0:
            logger.error("Training failed for %s:\n%s", config.name, result.stderr)
            raise RuntimeError(f"Training script failed for {config.name}: {result.stderr[-500:]}")

        logger.info("✅ Training complete for %s:\n%s", config.name, result.stdout[-500:])


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Evaluate new model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_new_model(**context) -> None:
    """
    Log that evaluation is complete.

    The champion/challenger comparison and auto-promotion to Production
    is handled automatically by mlflow_utils.py during training.
    This task just confirms training ran and logs the outcome.
    """
    logger.info(
        "Model evaluation complete. "
        "Champion/challenger comparison handled by MLflow registry. "
        "Check MLflow UI at localhost:5000 → Models → melanoma_classifier "
        "to see if new model was promoted to Production."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: Cleanup
# ─────────────────────────────────────────────────────────────────────────────

def cleanup(**context) -> None:
    """
    Clean up after retraining:
        - Remove retrain_needed.flag
        - Archive processed feedback images
    """
    # ── Remove retrain flag ───────────────────────────────────────────────
    if RETRAIN_FLAG_PATH.exists():
        RETRAIN_FLAG_PATH.unlink()
        logger.info("Retrain flag removed: %s", RETRAIN_FLAG_PATH)

    # ── Archive feedback data ─────────────────────────────────────────────
    archive_dir = FEEDBACK_DATA_DIR.parent / "feedback_data_archive" / \
                  datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)

    for class_name in ["malignant", "benign"]:
        class_dir = FEEDBACK_DATA_DIR / class_name
        if class_dir.exists() and any(class_dir.iterdir()):
            dest = archive_dir / class_name
            shutil.copytree(str(class_dir), str(dest))
            # Clear files inside directory but keep the directory itself
            # (avoids cross-container permission issues with Docker volumes)
            for f in class_dir.iterdir():
                try:
                    f.unlink()
                except PermissionError:
                    logger.warning("Could not delete %s — skipping", f)
            logger.info(
                "Archived feedback_data/%s → %s", class_name, archive_dir
            )

    logger.info("✅ Cleanup complete.")


# ── DAG definition ─────────────────────────────────────────────────────────────
default_args = {
    "owner":            "melanoma_mlops",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          0,   # no retries for retraining
}

with DAG(
    dag_id="melanoma_retraining_pipeline",
    description=(
        "Retraining pipeline triggered by misclassification rate or data drift. "
        "Merges feedback data, retrains model, auto-promotes if better."
    ),
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # manual trigger only
    catchup=False,
    tags=["melanoma", "retraining", "mlops"]
) as dag:

    t1 = PythonOperator(
        task_id="check_trigger",
        python_callable=check_trigger
    )

    t2 = PythonOperator(
        task_id="prepare_feedback_data",
        python_callable=prepare_feedback_data
    )

    t3 = PythonOperator(
        task_id="trigger_training",
        python_callable=trigger_training
    )

    t4 = PythonOperator(
        task_id="evaluate_new_model",
        python_callable=evaluate_new_model
    )

    t5 = PythonOperator(
        task_id="cleanup",
        python_callable=cleanup
    )

    t1 >> t2 >> t3 >> t4 >> t5
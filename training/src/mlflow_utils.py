"""
mlflow_utils.py
---------------
Phase 3 & 4 | Executed: Local (on-prem, CPU)

MLflow logging utilities for experiment tracking.

All tracking is local — no DagsHub, no cloud.
MLflow stores everything in the local mlruns/ folder.

MLflow UI can be opened any time with:
    mlflow ui --backend-store-uri ./mlruns
    → open http://localhost:5000

Custom logging beyond autolog:
    - confusion matrix JSON as artifact
    - classification report as artifact
    - best threshold value
    - data version tag
    - git commit hash tag
    - model parameter counts
    - per-epoch metrics (train + val)
    - baseline stats JSON as artifact
"""

import logging
import subprocess
from pathlib import Path

import mlflow
import mlflow.pytorch

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mlflow_utils")


def get_git_commit_hash() -> str:
    """
    Get the current Git commit hash for reproducibility tracking.

    Returns:
        str: Short git commit hash, or 'unknown' if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "unknown"
    except Exception:
        return "unknown"


# def setup_mlflow(config: dict) -> None:
#     """
#     Configure MLflow tracking URI and experiment.

#     Uses local mlruns/ folder for all tracking.
#     Creates the experiment if it does not exist.

#     Args:
#         config (dict): Parsed training config YAML.
#     """
#     # ── Set local tracking URI ────────────────────────────────────────────
#     mlruns_path = Path(config["mlflow"]["tracking_uri"])
#     mlruns_path.mkdir(parents=True, exist_ok=True)

#     mlflow.set_tracking_uri(str(mlruns_path))

#     # ── Set experiment ────────────────────────────────────────────────────
#     experiment_name = config["mlflow"]["experiment_name"]
#     mlflow.set_experiment(experiment_name)

#     logger.info(
#         "MLflow configured | tracking_uri=%s | experiment=%s",
#         mlruns_path, experiment_name
#     )

def setup_mlflow(config: dict) -> None:
    import os

    if os.environ.get("MLFLOW_RUN_ID"):
        # Running via mlflow run — only ensure experiment exists, touch nothing else
        client = mlflow.tracking.MlflowClient()
        experiment_name = config["mlflow"]["experiment_name"]
        if not client.get_experiment_by_name(experiment_name):
            client.create_experiment(experiment_name)
        logger.info("MLflow run detected — experiment ensured: %s", experiment_name)
        return

    # Running via python train.py directly
    mlruns_path = Path(config["mlflow"]["tracking_uri"])
    mlruns_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(mlruns_path))

    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    logger.info(
        "MLflow configured | tracking_uri=%s | experiment=%s",
        mlflow.get_tracking_uri(), experiment_name
    )
    
def log_config_params(config: dict) -> None:
    """
    Log all training configuration parameters to MLflow.

    Args:
        config (dict): Parsed training config YAML.
    """
    # ── Flatten config into MLflow params ─────────────────────────────────
    mlflow.log_params({
        "model_name": config["model"]["name"],
        "img_size": config["model"]["img_size"],
        "freeze_base": config["model"]["freeze_base"],
        "unfreeze_epoch": config["model"]["unfreeze_epoch"],
        "num_unfreeze": config["model"]["num_layers_to_unfreeze"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "lr_head": config["training"]["lr_head"],
        "lr_finetune": config["training"]["lr_finetune"],
        "weight_decay": config["training"]["weight_decay"],
        "early_stop": config["training"]["early_stopping_patience"],
        "data_version": config["data"]["version"],
        "debug_mode": config["debug"]["enabled"],
        "num_classes": 2
    })
    logger.info("Config params logged to MLflow")


def log_tags(config: dict, param_counts: dict) -> None:
    """
    Log custom tags to MLflow run for traceability.

    Tags logged:
        - git_commit_hash : ties this run to exact code version
        - data_version    : which dataset version was used
        - experiment_type : baseline / teacher / distillation
        - model_name      : architecture used
        - total_params    : total model parameters
        - trainable_params: trainable model parameters

    Args:
        config (dict): Parsed training config YAML.
        param_counts (dict): Output of model.count_parameters().
    """
    mlflow.set_tags({
        "git_commit_hash": get_git_commit_hash(),
        "data_version": config["data"]["version"],
        "experiment_type": config["mlflow"]["experiment_type"],
        "model_name": config["model"]["name"],
        "total_params": param_counts["total"],
        "trainable_params": param_counts["trainable"],
        "debug_mode": str(config["debug"]["enabled"])
    })
    logger.info("Tags logged to MLflow")


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_recall: float,
    train_f1: float,
    val_recall: float,
    val_f1: float,
    val_accuracy: float,
    val_auc: float
) -> None:
    """
    Log per-epoch metrics to MLflow.

    Args:
        epoch (int): Current epoch number.
        train_loss (float): Average training loss for this epoch.
        train_recall (float): Training recall.
        train_f1 (float): Training F1 score.
        val_recall (float): Validation recall.
        val_f1 (float): Validation F1 score.
        val_accuracy (float): Validation accuracy.
        val_auc (float): Validation AUC.
    """
    mlflow.log_metrics(
        {
            "train_loss": train_loss,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_accuracy": val_accuracy,
            "val_auc": val_auc
        },
        step=epoch
    )


def log_threshold(threshold: float) -> None:
    """
    Log the best auto-tuned classification threshold.

    Args:
        threshold (float): Best threshold from val set tuning.
    """
    mlflow.log_param("best_threshold", round(threshold, 4))
    logger.info("Best threshold logged: %.4f", threshold)


def log_artifacts(artifacts_dir: Path, baseline_stats_path: Path = None) -> None:
    """
    Log all evaluation artifacts to MLflow.

    Artifacts logged:
        - confusion matrix JSON
        - classification report TXT
        - baseline stats JSON (if path provided)

    Args:
        artifacts_dir (Path): Directory containing evaluation artifacts.
        baseline_stats_path (Path): Optional path to baseline stats JSON.
    """
    # ── Log evaluation artifacts ──────────────────────────────────────────
    for artifact_file in artifacts_dir.iterdir():
        if artifact_file.is_file():
            mlflow.log_artifact(str(artifact_file))
            logger.info("Artifact logged: %s", artifact_file.name)

    # ── Log baseline stats ────────────────────────────────────────────────
    if baseline_stats_path and Path(baseline_stats_path).exists():
        mlflow.log_artifact(str(baseline_stats_path), artifact_path="data")
        logger.info("Baseline stats logged: %s", baseline_stats_path)


def get_current_production_recall() -> float:
    """
    Get the val_recall of the current Production model from MLflow registry.

    Searches all runs that have a registered model version in Production stage
    and returns the best val_recall found.

    Returns:
        float: val_recall of current Production model, or 0.0 if none exists.
    """
    client = mlflow.tracking.MlflowClient()

    try:
        # Get all versions of the registered model in Production stage
        production_versions = client.get_latest_versions(
            name="melanoma_classifier",
            stages=["Production"]
        )

        if not production_versions:
            logger.info("No Production model found in registry. New model will be first.")
            return 0.0

        # Get the run associated with the current Production model
        production_run_id = production_versions[0].run_id
        client.get_run(production_run_id)

        # Extract val_recall from that run's metrics
        # MLflow stores per-step metrics — get the last value
        recall_history = client.get_metric_history(production_run_id, "val_recall")

        if not recall_history:
            logger.warning("Production model run has no val_recall metric. Returning 0.0")
            return 0.0

        best_recall = max(m.value for m in recall_history)
        logger.info(
            "Current Production model val_recall: %.4f (run_id: %s)",
            best_recall, production_run_id
        )
        return best_recall

    except Exception as e:
        logger.warning(
            "Could not fetch Production model recall: %s. Treating as 0.0", str(e)
        )
        return 0.0


def get_current_production_f1() -> float:
    """
    Get the val_f1 of the current Production model from MLflow registry.

    Returns:
        float: val_f1 of current Production model, or 0.0 if none exists.
    """
    client = mlflow.tracking.MlflowClient()

    try:
        production_versions = client.get_latest_versions(
            name="melanoma_classifier",
            stages=["Production"]
        )

        if not production_versions:
            return 0.0

        production_run_id = production_versions[0].run_id
        f1_history = client.get_metric_history(production_run_id, "val_f1")

        if not f1_history:
            logger.warning("Production model run has no val_f1 metric. Returning 0.0")
            return 0.0

        best_f1 = max(m.value for m in f1_history)
        logger.info(
            "Current Production model val_f1: %.4f (run_id: %s)",
            best_f1, production_run_id
        )
        return best_f1

    except Exception as e:
        logger.warning(
            "Could not fetch Production model f1: %s. Treating as 0.0", str(e)
        )
        return 0.0


def promote_to_production(model_version: int) -> None:
    """
    Promote a registered model version to Production stage.
    Archive any existing Production model.

    Args:
        model_version (int): The version number to promote.
    """
    client = mlflow.tracking.MlflowClient()

    # ── Archive current Production model first ────────────────────────────
    try:
        current_production = client.get_latest_versions(
            name="melanoma_classifier",
            stages=["Production"]
        )
        for version in current_production:
            client.transition_model_version_stage(
                name="melanoma_classifier",
                version=version.version,
                stage="Archived"
            )
            logger.info(
                "Archived previous Production model version: %s",
                version.version
            )
    except Exception as e:
        logger.warning("Could not archive old Production model: %s", str(e))

    # ── Promote new model to Production ──────────────────────────────────
    client.transition_model_version_stage(
        name="melanoma_classifier",
        version=model_version,
        stage="Production"
    )
    logger.info(
        "✅ Model version %d promoted to Production", model_version
    )


def log_model(
    model,
    model_name: str,
    best_checkpoint_path: Path,
    new_val_recall: float,
    new_val_f1: float = 0.0
) -> None:
    """
    Log model to MLflow registry and auto-promote to Production
    if it outperforms the current Production model (Champion/Challenger pattern).

    Two-stage comparison logic:
        Stage 1 — Recall comparison (primary metric, patient safety):
            If new_recall > current_recall + RECALL_TOLERANCE → promote
            If new_recall < current_recall - RECALL_TOLERANCE → keep current

        Stage 2 — F1 tiebreaker (when recall is tied within tolerance):
            If recalls are tied (within ±RECALL_TOLERANCE):
                If new_f1 > current_f1 → promote (better balance)
                Else → keep current (stability wins)

        This ensures:
            - Recall is always the primary safety metric
            - When models are equally safe, we pick the more precise one
            - Avoids promoting a model that flags everything as malignant

    Args:
        model: Trained PyTorch model.
        model_name (str): Architecture name.
        best_checkpoint_path (Path): Path to saved .pth checkpoint.
        new_val_recall (float): Best val_recall achieved in this training run.
        new_val_f1 (float): Best val_f1 achieved in this training run.
    """
        # ── Sanity check: reject degenerate models ────────────────────────────
    if new_val_f1 < 0.0:
        logger.warning(
            "Model has non-zero F1 (%.4f) "
            "Skipping registration and promotion.", new_val_f1
        )
        return
        
    RECALL_TOLERANCE = 0.04   # recalls within 4% are considered tied
    # ── Log raw .pth as artifact in this run ──────────────────────────────
    mlflow.log_artifact(str(best_checkpoint_path), artifact_path="model")
    logger.info(".pth checkpoint logged as artifact")

    # ── Register model in MLflow registry ────────────────────────────────
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="pytorch_model",
        registered_model_name="melanoma_classifier"
    )

    # Get the version number just created
    client = mlflow.tracking.MlflowClient()
    latest_versions = client.get_latest_versions(
        name="melanoma_classifier",
        stages=["None"]
    )
    new_version = latest_versions[0].version if latest_versions else 1

    logger.info(
        "Registered as melanoma_classifier version %s | val_recall=%.4f",
        new_version, new_val_recall
    )

    # ── Champion / Challenger comparison ─────────────────────────────────
    current_production_recall = get_current_production_recall()
    current_production_f1 = get_current_production_f1()

    recall_diff = new_val_recall - current_production_recall

    if recall_diff > RECALL_TOLERANCE:
        # Stage 1: New model has strictly better recall → promote
        logger.info(
            "Stage 1 PROMOTE: new recall (%.4f) > current (%.4f) by %.4f. "
            "Promoting to Production.",
            new_val_recall, current_production_recall, recall_diff
        )
        promote_to_production(new_version)
        mlflow.set_tag("promotion_result", "promoted_recall_improvement")

    elif recall_diff < -RECALL_TOLERANCE:
        # Stage 1: New model has worse recall → keep current
        logger.info(
            "Stage 1 KEEP: new recall (%.4f) < current (%.4f). "
            "Keeping current Production.",
            new_val_recall, current_production_recall
        )
        client.transition_model_version_stage(
            name="melanoma_classifier",
            version=new_version,
            stage="Staging"
        )
        mlflow.set_tag("promotion_result", "kept_recall_worse")

    else:
        # Stage 2: Recalls are tied — use F1 as tiebreaker
        logger.info(
            "Stage 2 (F1 tiebreaker): recalls tied (new=%.4f, current=%.4f). "
            "Comparing F1: new=%.4f vs current=%.4f.",
            new_val_recall, current_production_recall,
            new_val_f1, current_production_f1
        )

        if new_val_f1 > current_production_f1:
            logger.info(
                "Stage 2 PROMOTE: new F1 (%.4f) > current F1 (%.4f). "
                "Promoting to Production.",
                new_val_f1, current_production_f1
            )
            promote_to_production(new_version)
            mlflow.set_tag("promotion_result", "promoted_f1_tiebreaker")
        else:
            logger.info(
                "Stage 2 KEEP: new F1 (%.4f) <= current F1 (%.4f). "
                "Keeping current Production (stability wins).",
                new_val_f1, current_production_f1
            )
            client.transition_model_version_stage(
                name="melanoma_classifier",
                version=new_version,
                stage="Staging"
            )
            mlflow.set_tag("promotion_result", "kept_f1_not_better")

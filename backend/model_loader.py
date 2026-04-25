"""
model_loader.py
---------------
Phase 4 | Executed: Local (backend Docker container)

Loads the Production model from local MLflow registry at startup.
Caches model in memory — no reloading per request.

The mlruns/ folder is mounted as a Docker volume so the backend
always sees the latest Production model without rebuilding the container.

Auto-reload: if the Production model version changes (after retraining),
the backend detects it on the next health check and reloads.
"""

import os
import torch
import mlflow
import mlflow.pytorch
from pathlib import Path
from mlflow.tracking import MlflowClient

from logger import get_logger
from monitoring import MODEL_INFO, MODEL_LOAD_STATUS

logger = get_logger("model_loader")

# ── Registry config ───────────────────────────────────────────────────────────
REGISTERED_MODEL_NAME = "melanoma_classifier"
MODEL_STAGE            = "Production"

# ── Global model cache ────────────────────────────────────────────────────────
_model       = None
_model_meta  = {
    "version":    None,
    "name":       None,
    "run_id":     None,
    "threshold":  0.35    # default fallback threshold
}


def _get_mlruns_path() -> str:
    """
    Get the MLflow tracking URI from environment variable or default.

    Returns:
        str: Path to mlruns/ folder.
    """
    return os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")


def _get_production_version() -> tuple:
    """
    Get the current Production model version and run_id from registry.

    Returns:
        tuple: (version, run_id) or (None, None) if no Production model.
    """
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(
            name=REGISTERED_MODEL_NAME,
            stages=[MODEL_STAGE]
        )
        if not versions:
            return None, None
        v = versions[0]
        return v.version, v.run_id
    except Exception as e:
        logger.error("Failed to query MLflow registry: %s", str(e))
        return None, None


def _get_threshold_from_run(run_id: str) -> float:
    """
    Retrieve the best_threshold param logged during training for this run.

    Args:
        run_id (str): MLflow run ID.

    Returns:
        float: Threshold value, or 0.35 as fallback.
    """
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        threshold = run.data.params.get("best_threshold", None)
        if threshold is not None:
            return float(threshold)
        logger.warning("No best_threshold found in run %s. Using 0.35", run_id)
        return 0.35
    except Exception as e:
        logger.warning(
            "Could not fetch threshold from run %s: %s. Using 0.35",
            run_id, str(e)
        )
        return 0.35


def load_model() -> bool:
    """
    Load the Production model from MLflow registry into memory.

    Called once at FastAPI startup.
    Also called by the reload check when a new Production model is detected.

    Returns:
        bool: True if model loaded successfully, False otherwise.
    """
    global _model, _model_meta

    mlflow.set_tracking_uri(_get_mlruns_path())

    version, run_id = _get_production_version()

    if version is None:
        logger.error(
            "No Production model found in MLflow registry. "
            "Train a model first and ensure it gets promoted to Production."
        )
        MODEL_LOAD_STATUS.set(0)
        return False

    try:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        logger.info("Loading model from: %s (version %s)", model_uri, version)

        model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
        model.eval()

        # ── Fetch threshold from training run ─────────────────────────────
        threshold = _get_threshold_from_run(run_id)

        # ── Cache in memory ───────────────────────────────────────────────
        _model = model
        _model_meta["version"]   = str(version)
        _model_meta["run_id"]    = run_id
        _model_meta["threshold"] = threshold

        # ── Update Prometheus model info ──────────────────────────────────
        MODEL_INFO.info({
            "version":   str(version),
            "run_id":    run_id,
            "threshold": str(threshold),
            "stage":     MODEL_STAGE
        })
        MODEL_LOAD_STATUS.set(1)

        logger.info(
            "✅ Model loaded | version=%s | threshold=%.4f | run_id=%s",
            version, threshold, run_id
        )
        return True

    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        MODEL_LOAD_STATUS.set(0)
        return False


def get_model():
    """
    Get the cached model instance.

    Returns:
        torch.nn.Module or None: Loaded model, or None if not loaded.
    """
    return _model


def get_model_meta() -> dict:
    """
    Get metadata about the currently loaded model.

    Returns:
        dict: version, run_id, threshold, name.
    """
    return _model_meta.copy()


def check_and_reload() -> bool:
    """
    Check if a new Production model version is available.
    If so, reload the model automatically.

    Called periodically by the /health endpoint to enable
    zero-downtime model updates after retraining.

    Returns:
        bool: True if model was reloaded, False if no change.
    """
    current_version = _model_meta.get("version")
    latest_version, _ = _get_production_version()

    if latest_version is None:
        return False

    if str(latest_version) != str(current_version):
        logger.info(
            "New Production model detected (version %s → %s). Reloading...",
            current_version, latest_version
        )
        load_model()
        return True

    return False

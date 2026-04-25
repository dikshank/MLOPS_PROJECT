"""
model_loader.py
---------------
Phase 4 | Executed: Local (backend Docker container)

Loads the Production model from local MLflow registry at startup.
Caches model in memory — no reloading per request.

The mlruns/ folder is mounted as a Docker volume so the backend
always sees the latest Production model without rebuilding the container.

Key design: loads model directly from the mlruns/ folder using run_id
and artifact path — avoids Windows absolute path issues that occur when
training happens on Windows but serving happens in Linux Docker container.
"""

import os
import glob
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
_model      = None
_model_meta = {
    "version":   None,
    "name":      None,
    "run_id":    None,
    "threshold": 0.35
}


def _get_mlruns_path() -> str:
    """Get MLflow tracking URI from environment or default."""
    return os.environ.get("MLFLOW_TRACKING_URI", "/mlruns")


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


def _find_model_path(run_id: str, mlruns_path: str) -> str:
    """
    Find the pytorch_model directory directly in mlruns/ folder.

    This bypasses the Windows absolute path stored in MLflow artifacts
    by searching the mlruns/ folder directly using the run_id.

    Args:
        run_id (str): MLflow run ID.
        mlruns_path (str): Path to mlruns/ folder inside container.

    Returns:
        str: Path to the pytorch_model directory.

    Raises:
        FileNotFoundError: If model directory cannot be found.
    """
    # mlruns structure: mlruns/<experiment_id>/<run_id>/artifacts/pytorch_model
    pattern = str(Path(mlruns_path) / "*" / run_id / "artifacts" / "pytorch_model")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"Could not find pytorch_model for run_id={run_id} "
            f"in mlruns path={mlruns_path}. "
            f"Searched pattern: {pattern}"
        )

    model_path = matches[0]
    logger.info("Found model at: %s", model_path)
    return model_path


def _get_threshold_from_run(run_id: str) -> float:
    """
    Retrieve the best_threshold param logged during training.

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
        logger.warning("No best_threshold in run %s. Using 0.35", run_id)
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

    Uses run_id to find model directly in mlruns/ folder,
    bypassing any Windows absolute paths stored in MLflow metadata.

    Returns:
        bool: True if model loaded successfully, False otherwise.
    """
    global _model, _model_meta

    mlruns_path = _get_mlruns_path()
    mlflow.set_tracking_uri(mlruns_path)

    version, run_id = _get_production_version()

    if version is None:
        logger.error(
            "No Production model found in MLflow registry. "
            "Train a model first."
        )
        MODEL_LOAD_STATUS.set(0)
        return False

    try:
        # ── Find model directly in mlruns/ folder ─────────────────────────
        # This avoids Windows path issues in stored artifact URIs
        model_path = _find_model_path(run_id, mlruns_path)

        logger.info(
            "Loading model from local path: %s (version %s)",
            model_path, version
        )

        # ── Load PyTorch model ────────────────────────────────────────────
        model = mlflow.pytorch.load_model(model_path, map_location="cpu")
        model.eval()

        # ── Fetch threshold ───────────────────────────────────────────────
        threshold = _get_threshold_from_run(run_id)

        # ── Cache in memory ───────────────────────────────────────────────
        _model = model
        _model_meta["version"]   = str(version)
        _model_meta["run_id"]    = run_id
        _model_meta["threshold"] = threshold

        # ── Update Prometheus metrics ─────────────────────────────────────
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
    """Get the cached model instance."""
    return _model


def get_model_meta() -> dict:
    """Get metadata about the currently loaded model."""
    return _model_meta.copy()


def check_and_reload() -> bool:
    """
    Check if a new Production model version is available and reload if so.

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
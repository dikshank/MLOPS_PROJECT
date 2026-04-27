"""
main.py
-------
Phase 4 | Executed: Local (backend Docker container)

FastAPI backend for melanoma classification.

Endpoints:
    POST /predict   → upload image, get malignant/benign prediction
    GET  /health    → service liveness check
    GET  /ready     → model readiness check + auto-reload detection
    GET  /metrics   → Prometheus scrape endpoint
    POST /feedback  → submit ground truth label for real-world recall tracking

Key additions over base version:
    - Images saved at /predict time for retraining use
    - Drift detection on every prediction (vs baseline_stats.json)
    - Misclassification rate tracking
    - Automatic retraining trigger when thresholds exceeded
"""

import io
import os
import time
import json
import hashlib
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from logger import get_logger
from model_loader import load_model, get_model, get_model_meta, check_and_reload
from predictor import predict
from schemas import (
    PredictResponse,
    HealthResponse,
    ReadyResponse,
    FeedbackRequest,
    FeedbackResponse
)
from monitoring import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PREDICTION_COUNTER,
    CONFIDENCE_HISTOGRAM,
    MALIGNANT_PROBABILITY,
    ERROR_COUNTER,
    FEEDBACK_COUNTER,
    REAL_WORLD_RECALL,
    REAL_WORLD_PRECISION,
    MISCLASSIFICATION_COUNTER,
    MISCLASSIFICATION_RATE,
    DRIFT_SCORE,
    DRIFT_DETECTED,
    RETRAINING_TRIGGERED
)

logger = get_logger("main")

# ── Config from environment ───────────────────────────────────────────────────
# Thresholds for triggering retraining
MISCLASSIFICATION_THRESHOLD = float(
    os.environ.get("MISCLASSIFICATION_THRESHOLD", "0.10")
)
DRIFT_THRESHOLD = float(
    os.environ.get("DRIFT_THRESHOLD", "0.20")
)
# Minimum feedback count before misclassification rate is considered reliable
MIN_FEEDBACK_FOR_TRIGGER = int(
    os.environ.get("MIN_FEEDBACK_FOR_TRIGGER", "10")
)

# ── Paths ─────────────────────────────────────────────────────────────────────
LOGS_DIR = Path("logs")
FEEDBACK_LOG_PATH = LOGS_DIR / "feedback.jsonl"
RETRAINING_FLAG_PATH = LOGS_DIR / "retrain_needed.flag"

# Images saved at predict time — keyed by image_id
PENDING_FEEDBACK_DIR = LOGS_DIR / "pending_feedback"

# Images confirmed via feedback — organised by true label for retraining
FEEDBACK_DATA_DIR = LOGS_DIR / "feedback_data"

# Baseline stats from Airflow pipeline — used for drift detection
BASELINE_STATS_PATH = Path(
    os.environ.get("BASELINE_STATS_PATH", "data/baseline_stats/v1_stats.json")
)

# Create all required directories with open permissions
# so Airflow container can read/delete files written by backend container
for d in [LOGS_DIR, PENDING_FEEDBACK_DIR,
          FEEDBACK_DATA_DIR / "malignant",
          FEEDBACK_DATA_DIR / "benign"]:
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o777)

# ── In-memory state ───────────────────────────────────────────────────────────
_feedback_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
_baseline_histogram = None   # loaded once at startup


def _load_baseline_histogram() -> list:
    """
    Load pixel histogram from baseline_stats.json for drift detection.

    Returns:
        list: Normalised histogram counts, or None if file not found.
    """
    if not BASELINE_STATS_PATH.exists():
        logger.warning(
            "Baseline stats not found at %s. Drift detection disabled.",
            BASELINE_STATS_PATH
        )
        return None

    try:
        with open(BASELINE_STATS_PATH) as f:
            stats = json.load(f)
        counts = stats["pixel_histogram"]["counts"]
        total = sum(counts)
        # Normalise to proportions
        normalised = [c / total for c in counts] if total > 0 else counts
        logger.info("Baseline histogram loaded (%d bins)", len(normalised))
        return normalised
    except Exception as e:
        logger.warning("Could not load baseline histogram: %s", str(e))
        return None


def _compute_image_histogram(image_bytes: bytes, n_bins: int = 10) -> list:
    """
    Compute normalised pixel intensity histogram for an image.

    Args:
        image_bytes (bytes): Raw image bytes.
        n_bins (int): Number of histogram bins.

    Returns:
        list: Normalised histogram counts.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        counts, _ = np.histogram(arr.flatten(), bins=n_bins, range=(0.0, 1.0))
        total = sum(counts)
        return [c / total for c in counts] if total > 0 else list(counts)
    except Exception:
        return None


def _compute_drift_score(image_histogram: list) -> float:
    """
    Compute drift score as mean absolute difference between
    image histogram and baseline histogram.

    Args:
        image_histogram (list): Normalised histogram of incoming image.

    Returns:
        float: Drift score (0.0 = no drift, 1.0 = maximum drift).
    """
    if _baseline_histogram is None or image_histogram is None:
        return 0.0

    min_len = min(len(_baseline_histogram), len(image_histogram))
    diff = [abs(_baseline_histogram[i] - image_histogram[i])
            for i in range(min_len)]
    return float(np.mean(diff))


def _check_and_flag_retraining(reason: str) -> None:
    """
    Write a retraining flag file and update Prometheus counter.
    The Airflow retraining DAG polls this file to trigger retraining.

    Args:
        reason (str): 'misclassification' or 'drift'
    """
    flag_content = {
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feedback_counts": _feedback_counts.copy()
    }
    with open(RETRAINING_FLAG_PATH, "w") as f:
        json.dump(flag_content, f, indent=2)

    RETRAINING_TRIGGERED.labels(reason=reason).inc()
    logger.warning(
        "⚠️ RETRAINING FLAGGED — reason=%s | "
        "Flag written to: %s",
        reason, RETRAINING_FLAG_PATH
    )


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _baseline_histogram
    logger.info("FastAPI starting up — loading Production model...")
    success = load_model()
    if success:
        logger.info("✅ Model loaded successfully at startup")
    else:
        logger.error(
            "❌ Model failed to load at startup. "
            "/predict will return 503 until model is available."
        )
    _baseline_histogram = _load_baseline_histogram()
    yield
    logger.info("FastAPI shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Melanoma Detection API",
    description=(
        "Low-resolution melanoma detection for resource-constrained environments. "
        "Upload a skin lesion image to get a malignant/benign classification."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Classify an uploaded skin lesion image as malignant or benign.

    Also:
    - Saves image to pending_feedback/ for retraining use
    - Computes drift score vs baseline
    - Returns image_id for use with /feedback endpoint
    """
    start_time = time.time()

    model = get_model()
    if model is None:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="model_not_loaded").inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code="503").inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for the model to be ready."
        )

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="invalid_file_type").inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code="400").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG and PNG supported."
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        # ── Generate unique image_id ──────────────────────────────────────
        # Use hash of image bytes + timestamp for uniqueness
        image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        image_id = f"img_{image_hash}_{int(time.time())}.jpg"

        # ── Save image for potential retraining use ───────────────────────
        pending_path = PENDING_FEEDBACK_DIR / image_id
        with open(pending_path, "wb") as f:
            f.write(image_bytes)
        os.chmod(pending_path, 0o777)  # allow Airflow container to delete

        # ── Run inference ─────────────────────────────────────────────────
        meta = get_model_meta()
        result = predict(
            model=model,
            image_bytes=image_bytes,
            threshold=meta["threshold"]
        )

        # ── Drift detection ───────────────────────────────────────────────
        img_histogram = _compute_image_histogram(image_bytes)
        drift = _compute_drift_score(img_histogram)
        DRIFT_SCORE.set(drift)

        if drift > DRIFT_THRESHOLD:
            DRIFT_DETECTED.set(1)
            logger.warning(
                "Data drift detected | score=%.4f | threshold=%.4f",
                drift, DRIFT_THRESHOLD
            )
            _check_and_flag_retraining(reason="drift")
        else:
            DRIFT_DETECTED.set(0)

        # ── Update Prometheus metrics ─────────────────────────────────────
        PREDICTION_COUNTER.labels(label=result["label"]).inc()
        CONFIDENCE_HISTOGRAM.labels(label=result["label"]).observe(result["confidence"])
        MALIGNANT_PROBABILITY.observe(result["malignant_prob"])

        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code="200").inc()

        logger.info(
            "Prediction | image_id=%s | label=%s | latency=%.3fs | drift=%.4f",
            image_id, result["label"], latency, drift
        )

        return PredictResponse(image_id=image_id, **result)

    except ValueError as e:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="invalid_image").inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code="400").inc()
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="inference_error").inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="unexpected_error").inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code="500").inc()
        logger.error("Unexpected error in /predict: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status_code="200").inc()
    return HealthResponse(status="ok")


# ─────────────────────────────────────────────────────────────────────────────
# GET /ready
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/ready", response_model=ReadyResponse)
async def ready():
    reloaded = check_and_reload()
    if reloaded:
        logger.info("Model auto-reloaded at /ready check")

    model = get_model()
    meta = get_model_meta()

    REQUEST_COUNT.labels(endpoint="/ready", method="GET", status_code="200").inc()

    if model is None:
        return ReadyResponse(
            model_loaded=False,
            model_name=None,
            model_version=None,
            classification_threshold=None,
            status="not_ready"
        )

    return ReadyResponse(
        model_loaded=True,
        model_name=meta.get("name"),
        model_version=meta.get("version"),
        classification_threshold=meta.get("threshold") ,
        status="ready"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ─────────────────────────────────────────────────────────────────────────────
# POST /feedback
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """
    Submit ground truth label for a previous prediction.

    Also:
    - Moves saved image to feedback_data/<true_label>/ for retraining
    - Updates misclassification rate
    - Triggers retraining flag if rate exceeds threshold
    """

    try:
        predicted = request.predicted_label.lower()
        actual = request.true_label.lower()
        image_id = request.image_id

        valid_labels = {"malignant", "benign"}
        if predicted not in valid_labels or actual not in valid_labels:
            raise HTTPException(
                status_code=400,
                detail=f"Labels must be one of: {valid_labels}"
            )

        # ── Move image from pending to feedback_data ───────────────────────
        pending_path = PENDING_FEEDBACK_DIR / image_id
        feedback_path = FEEDBACK_DATA_DIR / actual / image_id

        if pending_path.exists():
            import shutil
            shutil.move(str(pending_path), str(feedback_path))
            os.chmod(feedback_path, 0o777)  # allow Airflow container to delete
            logger.info(
                "Image moved to feedback_data/%s/%s", actual, image_id
            )
        else:
            logger.warning(
                "Pending image not found for image_id=%s "
                "(may have expired or already processed)",
                image_id
            )

        # ── Update confusion matrix ───────────────────────────────────────
        if predicted == "malignant" and actual == "malignant":
            _feedback_counts["tp"] += 1
        elif predicted == "malignant" and actual == "benign":
            _feedback_counts["fp"] += 1
            MISCLASSIFICATION_COUNTER.labels(type="false_positive").inc()
        elif predicted == "benign" and actual == "malignant":
            _feedback_counts["fn"] += 1
            MISCLASSIFICATION_COUNTER.labels(type="false_negative").inc()
        else:
            _feedback_counts["tn"] += 1

        tp = _feedback_counts["tp"]
        fp = _feedback_counts["fp"]
        fn = _feedback_counts["fn"]
        tn = _feedback_counts["tn"]
        total_feedback = tp + fp + fn + tn

        # ── Update recall and precision gauges ────────────────────────────
        if (tp + fn) > 0:
            REAL_WORLD_RECALL.set(tp / (tp + fn))
        if (tp + fp) > 0:
            REAL_WORLD_PRECISION.set(tp / (tp + fp))

        # ── Compute and update misclassification rate ─────────────────────
        misclassifications = fp + fn
        if total_feedback > 0:
            rate = misclassifications / total_feedback
            MISCLASSIFICATION_RATE.set(rate)

            # ── Check retraining trigger ──────────────────────────────────
            if (total_feedback >= MIN_FEEDBACK_FOR_TRIGGER
                    and rate > MISCLASSIFICATION_THRESHOLD):
                logger.warning(
                    "Misclassification rate %.2f%% exceeds threshold %.2f%%",
                    rate * 100, MISCLASSIFICATION_THRESHOLD * 100
                )
                _check_and_flag_retraining(reason="misclassification")

        # ── Update feedback counter ───────────────────────────────────────
        FEEDBACK_COUNTER.labels(predicted=predicted, actual=actual).inc()

        # ── Log to JSONL ──────────────────────────────────────────────────
        log_entry = {
            "image_id": image_id,
            "predicted_label": predicted,
            "true_label": actual,
            "image_saved_to": str(feedback_path) if pending_path.exists() else None,
            "feedback_counts": _feedback_counts.copy(),
            "total_feedback": total_feedback
        }
        with open(FEEDBACK_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        REQUEST_COUNT.labels(endpoint="/feedback", method="POST", status_code="200").inc()

        logger.info(
            "Feedback | image=%s | predicted=%s | actual=%s | "
            "total_feedback=%d | misclassification_rate=%.4f",
            image_id, predicted, actual, total_feedback,
            (fp + fn) / total_feedback if total_feedback > 0 else 0.0
        )

        return FeedbackResponse(received=True, message="Feedback recorded. Thank you.")

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/feedback", error_type="unexpected_error").inc()
        logger.error("Error in /feedback: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error.")

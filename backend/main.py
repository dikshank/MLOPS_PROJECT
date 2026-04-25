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

Design:
    - Loosely coupled from frontend — all communication via REST API
    - Model loaded once at startup from MLflow local registry
    - Prometheus metrics updated on every request
    - Feedback endpoint enables real-world performance tracking
    - /ready endpoint triggers model reload if new Production version detected
"""

import time
import json
from pathlib import Path
from contextlib import asynccontextmanager

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
    REAL_WORLD_PRECISION
)

logger = get_logger("main")

# ── Feedback storage ──────────────────────────────────────────────────────────
# In-memory store for ground truth feedback
# Used to compute real-world recall for Grafana dashboard
FEEDBACK_LOG_PATH = Path("logs/feedback.jsonl")
FEEDBACK_LOG_PATH.parent.mkdir(exist_ok=True)

# Running confusion matrix counters for real-world recall
_feedback_counts = {
    "tp": 0,   # predicted malignant, actually malignant
    "fp": 0,   # predicted malignant, actually benign
    "fn": 0,   # predicted benign, actually malignant
    "tn": 0    # predicted benign, actually benign
}


# ── Lifespan: startup and shutdown ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("FastAPI starting up — loading Production model...")
    success = load_model()
    if success:
        logger.info("✅ Model loaded successfully at startup")
    else:
        logger.error(
            "❌ Model failed to load at startup. "
            "/predict will return 503 until model is available."
        )
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

# ── CORS: allow frontend (nginx on port 80) to call backend (port 8000) ───────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tightened in production
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

    Args:
        file: Uploaded image file (JPEG, PNG).

    Returns:
        PredictResponse: label, confidence, malignant_prob, threshold, recommendation.

    Raises:
        503: If model is not loaded.
        400: If file is not a valid image.
        500: If inference fails.
    """
    start_time = time.time()

    # ── Check model is loaded ─────────────────────────────────────────────
    model = get_model()
    if model is None:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="model_not_loaded").inc()
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="503"
        ).inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for the model to be ready."
        )

    # ── Validate file type ────────────────────────────────────────────────
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="invalid_file_type").inc()
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="400"
        ).inc()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Only JPEG and PNG are supported."
        )

    try:
        # ── Read image bytes ──────────────────────────────────────────────
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        # ── Run inference ─────────────────────────────────────────────────
        meta   = get_model_meta()
        result = predict(
            model=model,
            image_bytes=image_bytes,
            threshold=meta["threshold"]
        )

        # ── Update Prometheus metrics ─────────────────────────────────────
        PREDICTION_COUNTER.labels(label=result["label"]).inc()
        CONFIDENCE_HISTOGRAM.labels(label=result["label"]).observe(
            result["confidence"]
        )
        MALIGNANT_PROBABILITY.observe(result["malignant_prob"])

        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="200"
        ).inc()

        logger.info(
            "Prediction complete | label=%s | latency=%.3fs",
            result["label"], latency
        )

        return PredictResponse(**result)

    except ValueError as e:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="invalid_image").inc()
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="400"
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="inference_error").inc()
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="500"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="unexpected_error").inc()
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="500"
        ).inc()
        logger.error("Unexpected error in /predict: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Liveness check. Returns ok if the service is running.
    Used by Docker healthcheck and Airflow sensors.
    """
    REQUEST_COUNT.labels(
        endpoint="/health", method="GET", status_code="200"
    ).inc()
    return HealthResponse(status="ok")


# ─────────────────────────────────────────────────────────────────────────────
# GET /ready
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/ready", response_model=ReadyResponse)
async def ready():
    """
    Readiness check. Confirms model is loaded and ready to serve predictions.
    Also triggers auto-reload if a new Production model is detected in registry.
    """
    # ── Check for new Production model ────────────────────────────────────
    reloaded = check_and_reload()
    if reloaded:
        logger.info("Model auto-reloaded at /ready check")

    model = get_model()
    meta  = get_model_meta()

    REQUEST_COUNT.labels(
        endpoint="/ready", method="GET", status_code="200"
    ).inc()

    if model is None:
        return ReadyResponse(
            model_loaded=False,
            model_name=None,
            model_version=None,
            status="not_ready"
        )

    return ReadyResponse(
        model_loaded=True,
        model_name=meta.get("name"),
        model_version=meta.get("version"),
        status="ready"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics scrape endpoint.
    Returns all metrics in Prometheus text format.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /feedback
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """
    Submit ground truth label for a previous prediction.

    Enables real-world recall tracking — as doctors confirm diagnoses,
    the system computes how well the model is actually performing.

    This is the feedback loop required by the MLOps guidelines.
    """
    global _feedback_counts

    try:
        predicted = request.predicted_label.lower()
        actual    = request.true_label.lower()

        # ── Validate labels ───────────────────────────────────────────────
        valid_labels = {"malignant", "benign"}
        if predicted not in valid_labels or actual not in valid_labels:
            raise HTTPException(
                status_code=400,
                detail=f"Labels must be one of: {valid_labels}"
            )

        # ── Update confusion matrix counts ────────────────────────────────
        if predicted == "malignant" and actual == "malignant":
            _feedback_counts["tp"] += 1
        elif predicted == "malignant" and actual == "benign":
            _feedback_counts["fp"] += 1
        elif predicted == "benign" and actual == "malignant":
            _feedback_counts["fn"] += 1
        else:
            _feedback_counts["tn"] += 1

        # ── Update real-world recall gauge ────────────────────────────────
        tp = _feedback_counts["tp"]
        fn = _feedback_counts["fn"]
        fp = _feedback_counts["fp"]

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
            REAL_WORLD_RECALL.set(recall)

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
            REAL_WORLD_PRECISION.set(precision)

        # ── Update Prometheus feedback counter ────────────────────────────
        FEEDBACK_COUNTER.labels(
            predicted=predicted,
            actual=actual
        ).inc()

        # ── Log to JSONL file for audit trail ─────────────────────────────
        log_entry = {
            "image_id":        request.image_id,
            "predicted_label": predicted,
            "true_label":      actual,
            "feedback_counts": _feedback_counts.copy()
        }
        with open(FEEDBACK_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        REQUEST_COUNT.labels(
            endpoint="/feedback", method="POST", status_code="200"
        ).inc()

        logger.info(
            "Feedback received | image=%s | predicted=%s | actual=%s | "
            "running_recall=%.4f",
            request.image_id, predicted, actual,
            tp / (tp + fn) if (tp + fn) > 0 else 0.0
        )

        return FeedbackResponse(
            received=True,
            message="Feedback recorded. Thank you."
        )

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNTER.labels(
            endpoint="/feedback", error_type="unexpected_error"
        ).inc()
        logger.error("Error in /feedback: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error.")

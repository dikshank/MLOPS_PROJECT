# Low Level Design — API Endpoint Specifications

## 1. Base URL

```
http://localhost:8000
```

All endpoints are served by the FastAPI backend container.
The frontend connects via `window.API_BASE || "http://localhost:8000"`.

---

## 2. Endpoints

---

### POST /predict

**Description:** Classify an uploaded skin lesion image as malignant or benign.
Saves image to `logs/pending_feedback/` for potential retraining use.
Also performs drift detection against baseline pixel histogram.

**Request:**
```
Content-Type: multipart/form-data
Field: file (required) — image file, JPEG or PNG
```

**Response (200 OK):**
```json
{
  "image_id":       "img_a1b2c3d4_1714000000.jpg",
  "label":          "malignant",
  "confidence":     0.87,
  "malignant_prob": 0.87,
  "threshold_used": 0.35,
  "recommendation": "High risk detected. Please consult a dermatologist immediately."
}
```

**Error Responses:**
```
400 Bad Request        — invalid file type or empty file
422 Unprocessable      — missing file field
503 Service Unavailable — model not loaded
500 Internal Server Error — inference failure
```

**Processing Logic:**
```
1. Validate file type (JPEG or PNG only)
2. Read image bytes
3. Generate unique image_id (MD5 hash + timestamp)
4. Save image to logs/pending_feedback/<image_id>.jpg (for retraining)
5. Preprocess: resize to 64x64, normalize with ImageNet stats
6. Run inference through Production model
7. Apply recall-optimized threshold (auto-tuned during training)
8. Compute drift score vs baseline pixel histogram
9. If drift > 20% → write retrain_needed.flag
10. Update Prometheus metrics
11. Return PredictResponse including image_id
```

---

### GET /health

**Description:** Liveness check. Always returns 200 if service is running.
Used by Docker healthcheck and frontend status indicator.

**Request:** No parameters.

**Response (200 OK):**
```json
{
  "status": "ok"
}
```

---

### GET /ready

**Description:** Readiness check. Confirms model is loaded.
Also triggers auto-reload if a new Production model is detected in MLflow.

**Request:** No parameters.

**Response (200 OK) — model loaded:**
```json
{
  "model_loaded":   true,
  "model_name":     "simple_cnn",
  "model_version":  "13",
  "status":         "ready"
}
```

**Response (200 OK) — model not loaded:**
```json
{
  "model_loaded":   false,
  "model_name":     null,
  "model_version":  null,
  "status":         "not_ready"
}
```

---

### GET /metrics

**Description:** Prometheus metrics scrape endpoint.
Returns all metrics in Prometheus text exposition format.
Used by Prometheus scraper and Pipeline Dashboard tab.

**Request:** No parameters.

**Response (200 OK):**
```
Content-Type: text/plain; version=0.0.4; charset=utf-8

# HELP melanoma_request_total Total number of requests
# TYPE melanoma_request_total counter
melanoma_request_total{endpoint="/predict",method="POST",status_code="200"} 42.0
...
```

**Metrics exposed:**
```
melanoma_request_total           — requests per endpoint/method/status
melanoma_request_latency_seconds — latency histogram per endpoint
melanoma_prediction_total        — predictions by class label
melanoma_confidence_score        — confidence score histogram
melanoma_malignant_probability   — malignant probability histogram
melanoma_error_total             — errors per endpoint/type
melanoma_feedback_total          — feedback submissions by predicted/actual
melanoma_real_world_recall       — running recall from feedback (TP/(TP+FN))
melanoma_real_world_precision    — running precision from feedback (TP/(TP+FP))
melanoma_misclassification_total — FP + FN count
melanoma_misclassification_rate  — current misclassification rate
melanoma_drift_score             — pixel histogram MAE vs baseline
melanoma_drift_detected          — 1 if drift > threshold, else 0
melanoma_retraining_triggered    — count of retraining triggers
melanoma_model_loaded            — 1 if model loaded, 0 otherwise

```

---

### POST /feedback

**Description:** Submit ground truth label for a previous prediction.
Moves image from pending_feedback/ to feedback_data/<true_label>/.
Updates real-world recall, precision, misclassification rate.
Triggers retraining flag if misclassification rate exceeds threshold.

**Request:**
```
Content-Type: application/json
```
```json
{
  "image_id":        "img_a1b2c3d4_1714000000.jpg",
  "predicted_label": "malignant",
  "true_label":      "malignant"
}
```

**Field validation:**
```
image_id        — required, non-empty string (returned by /predict)
predicted_label — required, must be 'malignant' or 'benign'
true_label      — required, must be 'malignant' or 'benign'
```

**Response (200 OK):**
```json
{
  "received": true,
  "message":  "Feedback recorded. Thank you."
}
```

**Error Responses:**
```
400 Bad Request        — invalid label values
422 Unprocessable      — missing required fields
500 Internal Server Error — unexpected error
```

**Processing Logic:**
```
1. Validate predicted_label and true_label
2. Move image from logs/pending_feedback/ to logs/feedback_data/<true_label>/
3. Set file permissions to 0o777 (for cross-container Airflow access)
4. Update confusion matrix counts (TP, FP, FN, TN)
5. Recompute running recall = TP / (TP + FN)
6. Recompute running precision = TP / (TP + FP)
7. Compute misclassification rate = (FP + FN) / total
8. If total >= 10 and rate > 10% → write retrain_needed.flag
9. Update Prometheus gauges
10. Append to logs/feedback.jsonl audit log
```

---

## 3. Data Models

### PredictResponse
```python
image_id:       str    # unique ID for this prediction — pass to /feedback
label:          str    # 'malignant' or 'benign'
confidence:     float  # 0.0 to 1.0
malignant_prob: float  # 0.0 to 1.0
threshold_used: float  # 0.0 to 1.0
recommendation: str    # human-readable advice
```

### HealthResponse
```python
status: str    # always 'ok'
```

### ReadyResponse
```python
model_loaded:   bool           # True if model loaded
model_name:     str | None     # architecture name (e.g. 'simple_cnn')
model_version:  str | None     # MLflow registry version number
status:         str            # 'ready' or 'not_ready'
```

### FeedbackRequest
```python
image_id:        str   # returned by /predict endpoint
predicted_label: str   # 'malignant' or 'benign'
true_label:      str   # 'malignant' or 'benign'
```

### FeedbackResponse
```python
received: bool   # True if feedback recorded successfully
message:  str    # confirmation message
```

---

## 4. Inference Pipeline (Internal)

```
Input: raw image bytes (JPEG or PNG)
        |
PIL.Image.open() -> convert to RGB
        |
transforms.Resize((64, 64))
        |
transforms.ToTensor()
        |
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        |
unsqueeze(0) -> shape: (1, 3, 64, 64)
        |
model.forward(tensor) -> logits: (1, 2)
        |
softmax(logits) -> probs: (1, 2)
        |
malignant_prob = probs[0][1]
        |
if malignant_prob >= threshold -> label = 'malignant'
else                            -> label = 'benign'
        |
confidence = max(probs[0][0], probs[0][1])
        |
Output: PredictResponse (including image_id)
```

---

## 5. Drift Detection (Internal)

```
Input: raw image bytes
        |
Convert to numpy array, normalize to [0,1]
        |
Compute pixel intensity histogram (10 bins)
        |
Normalize histogram to proportions
        |
Compare with baseline histogram (from baseline_stats/v1_stats.json):
    drift_score = mean(|baseline[i] - image[i]|) for i in bins
        |
Update melanoma_drift_score gauge
        |
If drift_score > 0.20:
    melanoma_drift_detected = 1
    Write logs/retrain_needed.flag
    Increment melanoma_retraining_triggered counter
```

---

## 6. File Structure (Runtime)

```
logs/
  feedback.jsonl           # audit log of all feedback submissions
  retrain_needed.flag      # written when retraining threshold exceeded
  pending_feedback/        # images saved at /predict time
    img_<hash>_<ts>.jpg
  feedback_data/           # images confirmed via /feedback
    malignant/
      img_<hash>_<ts>.jpg
    benign/
      img_<hash>_<ts>.jpg
  feedback_data_archive/   # archived after each retraining run
    <timestamp>/
      malignant/
      benign/
```

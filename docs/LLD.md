# Low Level Design — API Endpoint Specifications

## 1. Base URL

```
http://localhost:8000
```

All endpoints are served by the FastAPI backend container.

---

## 2. Endpoints

---

### POST /predict

**Description:** Classify an uploaded skin lesion image as malignant or benign.

**Request:**
```
Content-Type: multipart/form-data
Field: file (required) — image file, JPEG or PNG
```

**Response (200 OK):**
```json
{
  "label":          "malignant",
  "confidence":     0.87,
  "malignant_prob": 0.87,
  "threshold_used": 0.35,
  "recommendation": "High risk detected. Please consult a dermatologist immediately."
}
```

**Error Responses:**
```
400 Bad Request   — invalid file type or empty file
422 Unprocessable — missing file field
503 Service Unavailable — model not loaded
500 Internal Server Error — inference failure
```

**Processing Logic:**
```
1. Validate file type (JPEG or PNG only)
2. Read image bytes
3. Preprocess: resize to 64×64, normalize with ImageNet stats
4. Run inference through Production model
5. Apply recall-optimized threshold (auto-tuned during training)
6. Return label, confidence, probability, threshold, recommendation
7. Update Prometheus metrics
```

---

### GET /health

**Description:** Liveness check. Always returns 200 if service is running.
Used by Docker healthcheck.

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
Also triggers auto-reload if a new Production model is detected.

**Request:** No parameters.

**Response (200 OK) — model loaded:**
```json
{
  "model_loaded":   true,
  "model_name":     "mobilenet_v3_small",
  "model_version":  "3",
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
melanoma_request_total          — requests per endpoint/method/status
melanoma_request_latency_seconds — latency histogram per endpoint
melanoma_prediction_total        — predictions by class label
melanoma_confidence_score        — confidence score histogram
melanoma_malignant_probability   — malignant probability histogram
melanoma_error_total             — errors per endpoint/type
melanoma_feedback_total          — feedback submissions by predicted/actual
melanoma_real_world_recall       — running recall from feedback
melanoma_real_world_precision    — running precision from feedback
melanoma_model_loaded            — 1 if model loaded, 0 otherwise
melanoma_model_info              — current model version and metadata
```

---

### POST /feedback

**Description:** Submit ground truth label for a previous prediction.
Enables real-world recall tracking and drift detection.

**Request:**
```
Content-Type: application/json
```
```json
{
  "image_id":        "img_20240101_001.jpg",
  "predicted_label": "malignant",
  "true_label":      "malignant"
}
```

**Field validation:**
```
image_id        — required, non-empty string
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
400 Bad Request   — invalid label values
422 Unprocessable — missing required fields
500 Internal Server Error — unexpected error
```

**Processing Logic:**
```
1. Validate predicted_label and true_label
2. Update confusion matrix counts (TP, FP, FN, TN)
3. Recompute running recall = TP / (TP + FN)
4. Recompute running precision = TP / (TP + FP)
5. Update Prometheus gauges
6. Append to feedback.jsonl audit log
```

---

## 3. Data Models

### PredictResponse
```python
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
model_name:     str | None     # architecture name
model_version:  str | None     # MLflow registry version
status:         str            # 'ready' or 'not_ready'
```

### FeedbackRequest
```python
image_id:        str   # identifier of the image
predicted_label: str   # 'malignant' or 'benign'
true_label:      str   # 'malignant' or 'benign'
```

### FeedbackResponse
```python
received: bool   # True if feedback recorded
message:  str    # confirmation message
```

---

## 4. Inference Pipeline (Internal)

```
Input: raw image bytes (JPEG or PNG)
        ↓
PIL.Image.open() → convert to RGB
        ↓
transforms.Resize((64, 64))
        ↓
transforms.ToTensor()
        ↓
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ↓
unsqueeze(0) → shape: (1, 3, 64, 64)
        ↓
model.forward(tensor) → logits: (1, 2)
        ↓
softmax(logits) → probs: (1, 2)
        ↓
malignant_prob = probs[0][1]
        ↓
if malignant_prob >= threshold → label = 'malignant'
else                           → label = 'benign'
        ↓
Output: PredictResponse
```

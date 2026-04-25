# Test Plan — Melanoma Detection MLOps System

## 1. Acceptance Criteria

The system meets acceptance criteria when ALL of the following are true:

| Criterion | Target | How Verified |
|---|---|---|
| val_recall (v2 model) | ≥ 0.80 | MLflow metrics |
| /predict latency | < 500ms on CPU | Grafana latency panel |
| Error rate | < 5% of requests | Grafana error rate panel |
| All unit tests pass | 100% pass rate | pytest test report |
| All Docker services start | No startup errors | docker-compose up |
| /health returns 200 | Always | test_health.py |
| /ready returns model_loaded: true | After training | test_health.py |
| Feedback loop functional | /feedback updates recall gauge | test_feedback.py |

---

## 2. Test Scope

### In Scope
- FastAPI endpoint behaviour (predict, health, ready, metrics, feedback)
- Airflow pipeline scripts (validate, split, preprocess, baseline_stats)
- Response schema validation
- Error handling and edge cases
- Prometheus metrics updates

### Out of Scope
- Model accuracy (evaluated separately via MLflow metrics)
- Grafana dashboard visual correctness
- Docker networking (verified manually)
- Browser compatibility

---

## 3. Test Cases

### 3.1 POST /predict

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| P01 | Valid JPEG upload | 64×64 JPEG | 200, valid schema |
| P02 | Valid PNG upload | 64×64 PNG | 200, valid schema |
| P03 | Tiny image (1×1px) | 1×1 JPEG | 200, no crash |
| P04 | Large image (1000×1000) | 1000×1000 JPEG | 200, resized correctly |
| P05 | Empty file | 0 bytes | 400 |
| P06 | Wrong file type | PDF bytes | 400 |
| P07 | Missing file field | No file | 422 |
| P08 | Model not loaded | Valid JPEG | 503 |
| P09 | Label validity | Valid JPEG | label ∈ {malignant, benign} |
| P10 | Confidence range | Valid JPEG | 0.0 ≤ confidence ≤ 1.0 |
| P11 | Probability range | Valid JPEG | 0.0 ≤ malignant_prob ≤ 1.0 |
| P12 | Threshold range | Valid JPEG | 0.0 ≤ threshold_used ≤ 1.0 |
| P13 | Recommendation non-empty | Valid JPEG | len(recommendation) > 0 |

### 3.2 GET /health

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| H01 | Health with model | — | 200, {status: ok} |
| H02 | Health without model | — | 200, {status: ok} |
| H03 | Status field present | — | "status" in response |

### 3.3 GET /ready

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| R01 | Ready with model | — | 200, model_loaded: true |
| R02 | Ready without model | — | 200, model_loaded: false |
| R03 | Status value valid | — | status ∈ {ready, not_ready} |
| R04 | model_loaded is bool | — | isinstance(model_loaded, bool) |
| R05 | Version present when loaded | — | model_version not None |

### 3.4 GET /metrics

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| M01 | Returns 200 | — | 200 |
| M02 | Content type correct | — | text/plain |
| M03 | Request counter present | — | melanoma_request_total in body |
| M04 | Model loaded gauge present | — | melanoma_model_loaded in body |
| M05 | Prediction counter present | — | melanoma_prediction_total in body |
| M06 | Counter increments after predict | After /predict | counter > 0 |

### 3.5 POST /feedback

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| F01 | TP feedback | malignant/malignant | 200, received: true |
| F02 | TN feedback | benign/benign | 200, received: true |
| F03 | FN feedback | benign/malignant | 200, received: true |
| F04 | FP feedback | malignant/benign | 200, received: true |
| F05 | Invalid predicted label | unknown/malignant | 400 |
| F06 | Invalid true label | malignant/maybe | 400 |
| F07 | Missing image_id | No image_id | 422 |
| F08 | Empty body | {} | 422 |
| F09 | Recall metric updates | After feedback | melanoma_real_world_recall in metrics |
| F10 | Multiple feedbacks | 4 feedbacks | All 200 |

### 3.6 Airflow Pipeline — validate.py

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| V01 | Valid dataset | Good images | No exception |
| V02 | Missing class folder | No benign/ | FileNotFoundError |
| V03 | Below min images | min=9999 | ValueError |
| V04 | Zero-byte file | 0-byte .jpg | Detected as corrupt |
| V05 | Returns summary dict | Good images | dict with class keys |
| V06 | Summary has required keys | Good images | total, valid, corrupt |

### 3.7 Airflow Pipeline — split.py

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| S01 | Flat structure detected | class/ at root | "flat" |
| S02 | Presplit structure detected | train/test/ exist | "presplit" |
| S03 | Unknown structure raises | Random folder | ValueError |
| S04 | Manifests created | Flat dataset | 3 CSV files |
| S05 | Manifests have correct columns | Any dataset | filepath, label |
| S06 | All images covered | Flat 20 images | total = 20 |
| S07 | Presplit test unchanged | 10 test images | test manifest = 10 |
| S08 | Labels are valid | Any dataset | labels ⊆ {malignant, benign} |

### 3.8 Airflow Pipeline — preprocess.py

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| PR01 | Output size correct | target 32×32 | all images 32×32 |
| PR02 | Output is RGB | Any input | all images RGB |
| PR03 | Folder structure created | Any dataset | train/val/test subfolders |

### 3.9 Airflow Pipeline — baseline_stats.py

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| B01 | JSON created | Processed data | stats.json exists |
| B02 | Required keys present | Any dataset | version, total, channels, histogram |
| B03 | Channel stats complete | Any dataset | R,G,B each have mean/std/min/max |
| B04 | Distribution sums correctly | Any dataset | sum(counts) == total_train_images |
| B05 | Histogram format correct | Any dataset | bin_edges length = counts length + 1 |

---

## 4. Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run only pipeline tests (no Docker required)
pytest tests/test_pipeline.py -v

# Run only API tests (requires Docker services running)
pytest tests/test_predict.py tests/test_health.py tests/test_feedback.py -v

# Run with coverage report
pytest tests/ --cov=airflow/scripts --cov=backend --cov-report=term-missing
```

---

## 5. Test Environment

```
OS:      Windows 11
Python:  3.10
pytest:  8.1.1
Docker:  required for API tests
```

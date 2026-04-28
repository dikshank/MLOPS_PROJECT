# Test Plan — Melanoma Detection MLOps System

## 1. Acceptance Criteria

The system meets acceptance criteria when ALL of the following are true:

| Criterion | Target | How Verified |
|---|---|---|
| /predict latency | < 200ms on CPU | Grafana latency panel |
| Error rate | < 5% of requests | Grafana error rate panel |
| All unit tests pass | 100% pass rate | pytest test report |
| All Docker services start | No startup errors | docker-compose up |
| /health returns 200 | Always | test_health.py |
| /ready returns model_loaded: true | After training | test_health.py |
| Feedback loop functional | /feedback updates recall gauge | test_feedback.py |
| Drift detection operational | drift_score updates on /predict | test_health.py |
| Retraining trigger operational | flag written at >10% misclassification | manual verification |

---

## 2. Test Scope

### In Scope
- FastAPI endpoint behaviour (predict, health, ready, metrics, feedback)
- Airflow pipeline scripts (validate, split, preprocess, baseline_stats)
- Response schema validation
- Error handling and edge cases
- Prometheus metrics updates
- image_id returned by /predict and accepted by /feedback

### Out of Scope
- Model accuracy (evaluated separately via MLflow metrics)
- Grafana dashboard visual correctness
- Docker networking (verified manually)
- Browser compatibility

---

## 3. Test Cases

### 3.1 POST /predict (15 tests)

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| P01 | Valid JPEG upload | 64x64 JPEG | 200, valid schema |
| P02 | Valid PNG upload | 64x64 PNG | 200, valid schema |
| P03 | Response has all required fields | Valid JPEG | image_id, label, confidence, malignant_prob, threshold_used, recommendation |
| P04 | Label validity | Valid JPEG | label in {malignant, benign} |
| P05 | Confidence range | Valid JPEG | 0.0 <= confidence <= 1.0 |
| P06 | Probability range | Valid JPEG | 0.0 <= malignant_prob <= 1.0 |
| P07 | Threshold range | Valid JPEG | 0.0 <= threshold_used <= 1.0 |
| P08 | Recommendation non-empty | Valid JPEG | len(recommendation) > 0 |
| P09 | Threshold matches model meta | Valid JPEG | threshold_used == 0.35 |
| P10 | Tiny image (1x1px) | 1x1 JPEG | 200, no crash |
| P11 | Large image (1000x1000) | 1000x1000 JPEG | 200, resized correctly |
| P12 | Wrong file type | PDF bytes | 400, "Invalid file type" |
| P13 | Empty file | 0 bytes | 400 |
| P14 | Missing file field | No file | 422 |
| P15 | Model not loaded | Valid JPEG | 503, "not loaded" |

### 3.2 GET /health (4 tests)

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| H01 | Health with model | — | 200, {status: ok} |
| H02 | Health without model | — | 200, {status: ok} |
| H03 | Status field present | — | "status" in response |
| H04 | Status value is ok | — | response == {status: ok} |

### 3.3 GET /ready (7 tests)

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| R01 | Ready with model | — | 200 |
| R02 | model_loaded is bool | — | isinstance(model_loaded, bool) |
| R03 | Status value valid | — | status in {ready, not_ready} |
| R04 | Returns ready when model loaded | — | model_loaded=True, status=ready |
| R05 | Returns not_ready without model | — | model_loaded=False, status=not_ready |
| R06 | Has required fields | — | model_loaded, status in response |
| R07 | Version present when loaded | — | model_version not None |

### 3.4 GET /metrics (6 tests)

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| M01 | Returns 200 | — | 200 |
| M02 | Content type correct | — | text/plain |
| M03 | Request counter present | — | melanoma_request_total in body |
| M04 | Model loaded gauge present | — | melanoma_model_loaded in body |
| M05 | Prediction counter present | — | melanoma_prediction_total in body |
| M06 | Counter increments after predict | After /predict | counter > 0 |

### 3.5 POST /feedback (14 tests)

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| F01 | TP feedback | malignant/malignant | 200 |
| F02 | TN feedback | benign/benign | 200 |
| F03 | FN feedback | benign/malignant | 200 |
| F04 | FP feedback | malignant/benign | 200 |
| F05 | Response has received field | Any feedback | "received" in response |
| F06 | received is True | Any feedback | received == True |
| F07 | Response has message | Any feedback | len(message) > 0 |
| F08 | Invalid predicted label | unknown/malignant | 400 |
| F09 | Invalid true label | malignant/maybe | 400 |
| F10 | Missing image_id | No image_id | 422 |
| F11 | Empty body | {} | 422 |
| F12 | Recall metric updates | After feedback | melanoma_real_world_recall in metrics |
| F13 | Multiple feedbacks accumulate | 4 feedbacks | All 200 |

### 3.6 Airflow Pipeline — validate.py (6 tests)

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| V01 | Valid dataset | Good images | No exception |
| V02 | Missing class folder | No benign/ | FileNotFoundError or ValueError |
| V03 | Below min images | min=9999 | ValueError "Minimum required" |
| V04 | Zero-byte file | 0-byte .jpg | Detected as corrupt, ValueError |
| V05 | Returns summary dict | Good images | dict with class keys |
| V06 | Summary has required keys | Good images | total, valid, corrupt per class |

### 3.7 Airflow Pipeline — split.py (8 tests)

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| S01 | Flat structure detected | class/ at root | "flat" |
| S02 | Presplit structure detected | train/test/ exist | "presplit" |
| S03 | Unknown structure raises | Random folder | ValueError |
| S04 | Manifests created | Flat dataset | 3 CSV files |
| S05 | Manifests have correct columns | Any dataset | filepath, label |
| S06 | All images covered | Flat 20 images | total = 20 |
| S07 | Presplit test unchanged | 10 test images | test manifest = 10 |
| S08 | Labels are valid | Any dataset | labels in {malignant, benign} |

### 3.8 Airflow Pipeline — preprocess.py (3 tests)

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| PR01 | Output size correct | target 32x32 | all images 32x32 |
| PR02 | Output is RGB | Any input | all images RGB mode |
| PR03 | Folder structure created | Any dataset | train/val/test subfolders |

### 3.9 Airflow Pipeline — baseline_stats.py (5 tests)

| ID | Test Case | Input | Expected Output |
|---|---|---|---|
| B01 | JSON created | Processed data | stats.json exists |
| B02 | Required keys present | Any dataset | version, total, channels, histogram |
| B03 | Channel stats complete | Any dataset | R,G,B each have mean/std/min/max/variance |
| B04 | Distribution sums correctly | Any dataset | sum(counts) == total_train_images |
| B05 | Histogram format correct | Any dataset | len(bin_edges) == len(counts) + 1 |

---

## 4. Test Execution

```bash
# Install test dependencies
pip install pytest pytest-md httpx python-multipart prometheus-client

# Run all 68 tests
pytest tests/ -v

# Generate markdown report
pytest tests/ -v --md=docs/test_report.md

# Run only pipeline tests (no Docker required)
pytest tests/test_pipeline.py -v

# Run only API tests (no Docker required — uses mocks)
pytest tests/test_predict.py tests/test_health.py tests/test_feedback.py -v

# Run with coverage report
pytest tests/ --cov=backend --cov-report=term-missing
```

---

## 5. Test Environment

```
OS:      Windows 11
Python:  3.14.3
pytest:  9.0.3
Docker:  not required (API tests use TestClient with mocks)
```

---

## 6. Test Summary

| Module | Tests |
|---|---|
| POST /predict | 15 |
| GET /health + /ready + /metrics | 17 |
| POST /feedback | 14 |
| Airflow pipeline scripts | 22 |
| **Total** | **68** |

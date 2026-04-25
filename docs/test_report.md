# Test Report — Melanoma Detection MLOps System

## Summary

| Metric | Value |
|---|---|
| Total Test Cases | 47 |
| Passed | *(fill after running)* |
| Failed | *(fill after running)* |
| Skipped | *(fill after running)* |
| Pass Rate | *(fill after running)* |
| Date Executed | *(fill after running)* |

---

## Acceptance Criteria Results

| Criterion | Target | Actual | Status |
|---|---|---|---|
| val_recall (v2 model) | ≥ 0.80 | *(fill)* | *(pass/fail)* |
| /predict latency | < 500ms | *(fill)* | *(pass/fail)* |
| Error rate | < 5% | *(fill)* | *(pass/fail)* |
| All unit tests | 100% pass | *(fill)* | *(pass/fail)* |
| Docker services start | No errors | *(fill)* | *(pass/fail)* |

---

## Test Results by Module

### POST /predict (13 tests)
| ID | Test Case | Status | Notes |
|---|---|---|---|
| P01 | Valid JPEG upload | | |
| P02 | Valid PNG upload | | |
| P03 | Tiny image (1×1px) | | |
| P04 | Large image (1000×1000) | | |
| P05 | Empty file | | |
| P06 | Wrong file type | | |
| P07 | Missing file field | | |
| P08 | Model not loaded | | |
| P09 | Label validity | | |
| P10 | Confidence range | | |
| P11 | Probability range | | |
| P12 | Threshold range | | |
| P13 | Recommendation non-empty | | |

### GET /health + /ready + /metrics (12 tests)
| ID | Test Case | Status | Notes |
|---|---|---|---|
| H01-H03 | Health tests | | |
| R01-R05 | Ready tests | | |
| M01-M06 | Metrics tests | | |

### POST /feedback (10 tests)
| ID | Test Case | Status | Notes |
|---|---|---|---|
| F01-F10 | Feedback tests | | |

### Airflow Pipeline (12 tests)
| ID | Test Case | Status | Notes |
|---|---|---|---|
| V01-V06 | validate.py tests | | |
| S01-S08 | split.py tests | | |
| PR01-PR03 | preprocess.py tests | | |
| B01-B05 | baseline_stats.py tests | | |

---

## Pytest Output

```
*(paste pytest output here after running)*
```

---

## Notes and Known Issues

*(fill after running tests)*

# Test Report — Melanoma Detection MLOps System

## Acceptance Criteria Results

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| All unit tests pass | 100% | 100% (68/68) | ✅ PASS |
| /predict latency | < 200ms | ~150ms avg | ✅ PASS |
| Error rate | < 5% | 0% | ✅ PASS |
| Docker services start | No errors | All 5 services up | ✅ PASS |
| Model loaded at startup | Yes | version=13 (SimpleCNN) | ✅ PASS |
| Drift detection | Operational | score=0.11 (threshold=0.20) | ✅ PASS |
| Retraining trigger | Operational | triggers at >10% misclass | ✅ PASS |

**All acceptance criteria met. ✅**

---

## Summary

| Metric | Value |
|--------|-------|
| Total Test Cases | 68 |
| Passed | 68 |
| Failed | 0 |
| Skipped | 0 |
| Pass Rate | 100% |
| Execution Time | 32.98 seconds |
| Date Executed | 26-Apr-2026 |
| Python Version | 3.14.3 |
| pytest Version | 9.0.3 |

---

## Test Coverage by Module

| Module | Tests | Passed | Failed |
|--------|-------|--------|--------|
| POST /feedback | 14 | 14 | 0 |
| GET /health + /ready + /metrics | 17 | 17 | 0 |
| Airflow pipeline scripts | 22 | 22 | 0 |
| POST /predict | 15 | 15 | 0 |
| **Total** | **68** | **68** | **0** |

---

## Detailed Results

*Report generated on 26-Apr-2026 at 16:10:29 by [pytest-md]*

[pytest-md]: https://github.com/hackebrot/pytest-md

### tests/test_feedback.py (14 tests)

| Test | Time | Status |
|------|------|--------|
| test_valid_feedback_malignant_malignant_returns_200 | 0.01s | ✅ |
| test_valid_feedback_benign_benign_returns_200 | 0.01s | ✅ |
| test_valid_feedback_false_negative_returns_200 | 0.01s | ✅ |
| test_valid_feedback_false_positive_returns_200 | 0.01s | ✅ |
| test_feedback_response_has_received_field | 0.01s | ✅ |
| test_feedback_received_is_true | 0.01s | ✅ |
| test_feedback_response_has_message | 0.01s | ✅ |
| test_invalid_predicted_label_returns_400 | 0.00s | ✅ |
| test_invalid_true_label_returns_400 | 0.00s | ✅ |
| test_missing_image_id_returns_422 | 0.01s | ✅ |
| test_empty_body_returns_422 | 0.00s | ✅ |
| test_metrics_update_after_feedback | 0.01s | ✅ |
| test_real_world_recall_metric_present | 0.02s | ✅ |
| test_multiple_feedbacks_accumulate | 0.04s | ✅ |

### tests/test_health.py (17 tests)

| Test | Time | Status |
|------|------|--------|
| test_health_returns_200 | 0.00s | ✅ |
| test_health_returns_ok_status | 0.00s | ✅ |
| test_health_returns_200_even_without_model | 0.00s | ✅ |
| test_health_has_status_field | 0.00s | ✅ |
| test_ready_returns_200 | 0.50s | ✅ |
| test_ready_model_loaded_is_bool | 0.36s | ✅ |
| test_ready_status_is_valid_value | 0.55s | ✅ |
| test_ready_returns_ready_when_model_loaded | 0.41s | ✅ |
| test_ready_returns_not_ready_when_no_model | 0.61s | ✅ |
| test_ready_has_required_fields | 0.41s | ✅ |
| test_ready_model_version_present_when_loaded | 0.47s | ✅ |
| test_metrics_returns_200 | 0.00s | ✅ |
| test_metrics_content_type_is_prometheus | 0.00s | ✅ |
| test_metrics_contains_request_counter | 0.01s | ✅ |
| test_metrics_contains_model_loaded_gauge | 0.01s | ✅ |
| test_metrics_contains_prediction_counter | 0.00s | ✅ |
| test_metrics_updates_after_prediction | 0.09s | ✅ |

### tests/test_pipeline.py (22 tests)

| Test | Time | Status |
|------|------|--------|
| test_valid_dataset_passes | 0.09s | ✅ |
| test_missing_class_folder_raises | 0.01s | ✅ |
| test_below_min_images_raises | 0.04s | ✅ |
| test_zero_byte_file_detected | 0.03s | ✅ |
| test_returns_summary_dict | 0.05s | ✅ |
| test_summary_has_expected_keys | 0.07s | ✅ |
| test_flat_structure_detected | 0.00s | ✅ |
| test_presplit_structure_detected | 0.00s | ✅ |
| test_unknown_structure_raises | 0.00s | ✅ |
| test_flat_split_creates_manifests | 0.08s | ✅ |
| test_manifests_have_correct_columns | 0.16s | ✅ |
| test_flat_split_covers_all_images | 0.09s | ✅ |
| test_presplit_test_size_unchanged | 0.04s | ✅ |
| test_labels_are_valid | 0.11s | ✅ |
| test_output_images_are_correct_size | 0.32s | ✅ |
| test_output_images_are_rgb | 0.29s | ✅ |
| test_output_folder_structure_created | 0.39s | ✅ |
| test_stats_json_created | 0.42s | ✅ |
| test_stats_has_required_top_level_keys | 0.45s | ✅ |
| test_channel_stats_have_required_fields | 0.41s | ✅ |
| test_class_distribution_sums_correctly | 0.32s | ✅ |
| test_histogram_has_bin_edges_and_counts | 0.33s | ✅ |

### tests/test_predict.py (15 tests)

| Test | Time | Status |
|------|------|--------|
| test_valid_jpeg_returns_200 | 0.01s | ✅ |
| test_valid_png_returns_200 | 0.01s | ✅ |
| test_response_has_all_required_fields | 0.01s | ✅ |
| test_label_is_valid | 0.01s | ✅ |
| test_confidence_in_range | 0.01s | ✅ |
| test_malignant_prob_in_range | 0.01s | ✅ |
| test_threshold_used_in_range | 0.01s | ✅ |
| test_recommendation_is_non_empty | 0.01s | ✅ |
| test_threshold_matches_model_meta | 0.01s | ✅ |
| test_tiny_1x1_image_does_not_crash | 0.01s | ✅ |
| test_large_1000x1000_image_returns_200 | 0.11s | ✅ |
| test_wrong_file_type_returns_400 | 0.01s | ✅ |
| test_empty_file_returns_400 | 0.00s | ✅ |
| test_missing_file_field_returns_422 | 0.01s | ✅ |
| test_model_not_loaded_returns_503 | 0.00s | ✅ |

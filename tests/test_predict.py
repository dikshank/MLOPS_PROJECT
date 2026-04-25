"""
test_predict.py
---------------
Phase 5 | Executed: Local

Unit tests for POST /predict endpoint.

Tests cover:
    - Valid JPEG upload → 200 + correct response schema
    - Valid PNG upload → 200 + correct response schema
    - Empty file → 400
    - Wrong file type (PDF) → 400
    - Missing file field → 422
    - Very small image (1x1px) → 200 (edge case, should not crash)
    - Large image (1000x1000) → 200 (resized correctly)
    - Model not loaded → 503
    - Response schema validation (all fields present and valid)
    - Label is always 'malignant' or 'benign'
    - Confidence is between 0 and 1
    - malignant_prob is between 0 and 1
    - threshold_used matches model meta threshold
    - recommendation is non-empty string
"""

import io
import pytest


VALID_LABELS     = {"malignant", "benign"}
PREDICT_ENDPOINT = "/predict"


# ── Helper ────────────────────────────────────────────────────────────────────

def upload_image(client, image_bytes: bytes, content_type: str = "image/jpeg"):
    """Send a POST /predict request with given image bytes."""
    return client.post(
        PREDICT_ENDPOINT,
        files={"file": ("test_image.jpg", io.BytesIO(image_bytes), content_type)}
    )


# ── Happy path tests ──────────────────────────────────────────────────────────

class TestPredictHappyPath:

    def test_valid_jpeg_returns_200(self, client_with_model, valid_jpeg):
        """Valid JPEG upload should return 200."""
        response = upload_image(client_with_model, valid_jpeg, "image/jpeg")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_valid_png_returns_200(self, client_with_model, valid_png):
        """Valid PNG upload should return 200."""
        response = upload_image(client_with_model, valid_png, "image/png")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_response_has_all_required_fields(self, client_with_model, valid_jpeg):
        """Response must contain all schema fields."""
        response = upload_image(client_with_model, valid_jpeg)
        assert response.status_code == 200
        data = response.json()

        required_fields = {
            "label", "confidence", "malignant_prob",
            "threshold_used", "recommendation"
        }
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_label_is_valid(self, client_with_model, valid_jpeg):
        """Label must be 'malignant' or 'benign'."""
        response = upload_image(client_with_model, valid_jpeg)
        assert response.status_code == 200
        assert response.json()["label"] in VALID_LABELS

    def test_confidence_in_range(self, client_with_model, valid_jpeg):
        """Confidence must be between 0.0 and 1.0."""
        response = upload_image(client_with_model, valid_jpeg)
        assert response.status_code == 200
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"

    def test_malignant_prob_in_range(self, client_with_model, valid_jpeg):
        """Malignant probability must be between 0.0 and 1.0."""
        response = upload_image(client_with_model, valid_jpeg)
        assert response.status_code == 200
        prob = response.json()["malignant_prob"]
        assert 0.0 <= prob <= 1.0, f"Malignant prob out of range: {prob}"

    def test_threshold_used_in_range(self, client_with_model, valid_jpeg):
        """Threshold used must be between 0.0 and 1.0."""
        response = upload_image(client_with_model, valid_jpeg)
        assert response.status_code == 200
        threshold = response.json()["threshold_used"]
        assert 0.0 <= threshold <= 1.0, f"Threshold out of range: {threshold}"

    def test_recommendation_is_non_empty(self, client_with_model, valid_jpeg):
        """Recommendation must be a non-empty string."""
        response = upload_image(client_with_model, valid_jpeg)
        assert response.status_code == 200
        rec = response.json()["recommendation"]
        assert isinstance(rec, str) and len(rec) > 0

    def test_threshold_matches_model_meta(self, client_with_model, valid_jpeg):
        """Threshold used must match the model's loaded threshold (0.35 in fixture)."""
        response = upload_image(client_with_model, valid_jpeg)
        assert response.status_code == 200
        assert response.json()["threshold_used"] == pytest.approx(0.35, abs=0.01)


# ── Edge case tests ───────────────────────────────────────────────────────────

class TestPredictEdgeCases:

    def test_tiny_1x1_image_does_not_crash(self, client_with_model, tiny_image):
        """1x1 pixel image should be handled without crashing (resized to 64x64)."""
        response = upload_image(client_with_model, tiny_image)
        assert response.status_code == 200

    def test_large_1000x1000_image_returns_200(self, client_with_model, large_image):
        """Large image should be resized and processed correctly."""
        response = upload_image(client_with_model, large_image)
        assert response.status_code == 200


# ── Error case tests ──────────────────────────────────────────────────────────

class TestPredictErrorCases:

    def test_wrong_file_type_returns_400(self, client_with_model, pdf_bytes):
        """Non-image file type should return 400."""
        response = upload_image(
            client_with_model, pdf_bytes, "application/pdf"
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_empty_file_returns_400(self, client_with_model, empty_bytes):
        """Empty file should return 400."""
        response = upload_image(client_with_model, empty_bytes, "image/jpeg")
        assert response.status_code == 400

    def test_missing_file_field_returns_422(self, client_with_model):
        """Request with no file field should return 422 (validation error)."""
        response = client_with_model.post(PREDICT_ENDPOINT)
        assert response.status_code == 422

    def test_model_not_loaded_returns_503(self, client_no_model, valid_jpeg):
        """If model is not loaded, /predict should return 503."""
        response = upload_image(client_no_model, valid_jpeg)
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"].lower()

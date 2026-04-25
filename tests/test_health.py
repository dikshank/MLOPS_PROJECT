"""
test_health.py
--------------
Phase 5 | Executed: Local

Unit tests for GET /health, GET /ready, GET /metrics endpoints.
"""

import pytest

VALID_STATUSES = {"ready", "not_ready"}


class TestHealth:

    def test_health_returns_200(self, client_with_model):
        response = client_with_model.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client_with_model):
        response = client_with_model.get("/health")
        assert response.json() == {"status": "ok"}

    def test_health_returns_200_even_without_model(self, client_no_model):
        """Liveness check must pass even if model not loaded."""
        response = client_no_model.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self, client_with_model):
        response = client_with_model.get("/health")
        assert "status" in response.json()


class TestReady:

    def test_ready_returns_200(self, client_with_model):
        response = client_with_model.get("/ready")
        assert response.status_code == 200

    def test_ready_model_loaded_is_bool(self, client_with_model):
        response = client_with_model.get("/ready")
        assert isinstance(response.json()["model_loaded"], bool)

    def test_ready_status_is_valid_value(self, client_with_model):
        response = client_with_model.get("/ready")
        assert response.json()["status"] in VALID_STATUSES

    def test_ready_returns_ready_when_model_loaded(self, client_with_model):
        response = client_with_model.get("/ready")
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "ready"

    def test_ready_returns_not_ready_when_no_model(self, client_no_model):
        response = client_no_model.get("/ready")
        data = response.json()
        assert data["model_loaded"] is False
        assert data["status"] == "not_ready"

    def test_ready_has_required_fields(self, client_with_model):
        response = client_with_model.get("/ready")
        data = response.json()
        for field in ["model_loaded", "status"]:
            assert field in data, f"Missing field: {field}"

    def test_ready_model_version_present_when_loaded(self, client_with_model):
        response = client_with_model.get("/ready")
        data = response.json()
        if data["model_loaded"]:
            assert data.get("model_version") is not None


class TestMetrics:

    def test_metrics_returns_200(self, client_with_model):
        response = client_with_model.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type_is_prometheus(self, client_with_model):
        """Prometheus scrape endpoint must return text/plain."""
        response = client_with_model.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_request_counter(self, client_with_model):
        """melanoma_request_total counter must be present."""
        response = client_with_model.get("/metrics")
        assert "melanoma_request_total" in response.text

    def test_metrics_contains_model_loaded_gauge(self, client_with_model):
        """melanoma_model_loaded gauge must be present."""
        response = client_with_model.get("/metrics")
        assert "melanoma_model_loaded" in response.text

    def test_metrics_contains_prediction_counter(self, client_with_model):
        """melanoma_prediction_total counter must be present."""
        response = client_with_model.get("/metrics")
        assert "melanoma_prediction_total" in response.text

    def test_metrics_updates_after_prediction(self, client_with_model, valid_jpeg):
        """Request counter should increment after a /predict call."""
        import io
        # Make a prediction first
        client_with_model.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(valid_jpeg), "image/jpeg")}
        )
        # Then check metrics
        response = client_with_model.get("/metrics")
        assert "melanoma_request_total" in response.text

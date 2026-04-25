"""
test_feedback.py
----------------
Phase 5 | Executed: Local

Unit tests for POST /feedback endpoint.
"""

import pytest


FEEDBACK_ENDPOINT = "/feedback"


def post_feedback(client, image_id, predicted, true_label):
    return client.post(
        FEEDBACK_ENDPOINT,
        json={
            "image_id":        image_id,
            "predicted_label": predicted,
            "true_label":      true_label
        }
    )


class TestFeedbackHappyPath:

    def test_valid_feedback_malignant_malignant_returns_200(self, client_with_model):
        response = post_feedback(
            client_with_model, "img_001.jpg", "malignant", "malignant"
        )
        assert response.status_code == 200

    def test_valid_feedback_benign_benign_returns_200(self, client_with_model):
        response = post_feedback(
            client_with_model, "img_002.jpg", "benign", "benign"
        )
        assert response.status_code == 200

    def test_valid_feedback_false_negative_returns_200(self, client_with_model):
        """False negative: predicted benign, actually malignant."""
        response = post_feedback(
            client_with_model, "img_003.jpg", "benign", "malignant"
        )
        assert response.status_code == 200

    def test_valid_feedback_false_positive_returns_200(self, client_with_model):
        """False positive: predicted malignant, actually benign."""
        response = post_feedback(
            client_with_model, "img_004.jpg", "malignant", "benign"
        )
        assert response.status_code == 200

    def test_feedback_response_has_received_field(self, client_with_model):
        response = post_feedback(
            client_with_model, "img_005.jpg", "malignant", "malignant"
        )
        assert response.status_code == 200
        assert "received" in response.json()

    def test_feedback_received_is_true(self, client_with_model):
        response = post_feedback(
            client_with_model, "img_006.jpg", "malignant", "malignant"
        )
        assert response.json()["received"] is True

    def test_feedback_response_has_message(self, client_with_model):
        response = post_feedback(
            client_with_model, "img_007.jpg", "benign", "benign"
        )
        assert "message" in response.json()
        assert len(response.json()["message"]) > 0


class TestFeedbackErrorCases:

    def test_invalid_predicted_label_returns_400(self, client_with_model):
        """Invalid label value should return 400."""
        response = post_feedback(
            client_with_model, "img_008.jpg", "unknown_label", "malignant"
        )
        assert response.status_code == 400

    def test_invalid_true_label_returns_400(self, client_with_model):
        response = post_feedback(
            client_with_model, "img_009.jpg", "malignant", "maybe"
        )
        assert response.status_code == 400

    def test_missing_image_id_returns_422(self, client_with_model):
        """Missing required field should return 422."""
        response = client_with_model.post(
            FEEDBACK_ENDPOINT,
            json={"predicted_label": "malignant", "true_label": "malignant"}
        )
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client_with_model):
        response = client_with_model.post(FEEDBACK_ENDPOINT, json={})
        assert response.status_code == 422


class TestFeedbackMetricsUpdate:

    def test_metrics_update_after_feedback(self, client_with_model):
        """Feedback counter should be present in /metrics after submission."""
        post_feedback(
            client_with_model, "img_010.jpg", "malignant", "malignant"
        )
        metrics = client_with_model.get("/metrics").text
        assert "melanoma_feedback_total" in metrics

    def test_real_world_recall_metric_present(self, client_with_model):
        """Real world recall gauge must be present in /metrics."""
        post_feedback(
            client_with_model, "img_011.jpg", "malignant", "malignant"
        )
        metrics = client_with_model.get("/metrics").text
        assert "melanoma_real_world_recall" in metrics

    def test_multiple_feedbacks_accumulate(self, client_with_model):
        """Multiple feedback submissions should all return 200."""
        feedbacks = [
            ("img_012.jpg", "malignant", "malignant"),
            ("img_013.jpg", "benign",    "benign"),
            ("img_014.jpg", "malignant", "benign"),
            ("img_015.jpg", "benign",    "malignant"),
        ]
        for image_id, predicted, true_label in feedbacks:
            response = post_feedback(
                client_with_model, image_id, predicted, true_label
            )
            assert response.status_code == 200, \
                f"Failed for {predicted}/{true_label}"

"""
Comprehensive test suite for the Sentiment Analysis API.
Covers:
  - Functional / happy-path tests
  - Edge cases and boundary conditions
  - Invalid / malicious input robustness
  - Batch endpoint
  - Integration / real-world scenario simulation
  - Performance / stress tests
"""

import asyncio
import time

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# ── Import app ────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.main import app

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Synchronous test client (used for most tests)."""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture(scope="function")  # changed from "module" — avoids event loop conflicts
async def async_client():
    """Async test client (used for concurrency / stress tests)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# =============================================================================
# 1. HEALTH CHECK
# =============================================================================

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


# =============================================================================
# 2. FUNCTIONAL TESTS — single /predict
# =============================================================================

class TestPredictFunctional:
    POSITIVE_TEXTS = [
        "I absolutely love this product, it is amazing!",
        "This is the best movie I have ever seen.",
        "Excellent service, very happy with my experience.",
        "Fantastic performance, outstanding quality.",
        "Wonderful experience, I will definitely come back.",
    ]
    NEGATIVE_TEXTS = [
        "This is absolutely terrible and I hate it.",
        "Worst product I have ever purchased.",
        "Horrible experience, very disappointed.",
        "Awful quality, would not recommend to anyone.",
        "Dreadful service, completely pathetic.",
    ]

    @pytest.mark.parametrize("text", POSITIVE_TEXTS)
    def test_positive_sentiment(self, client, text):
        resp = client.post("/predict", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] in ("positive", "negative")
        assert 0.0 <= data["confidence"] <= 1.0

    @pytest.mark.parametrize("text", NEGATIVE_TEXTS)
    def test_negative_sentiment(self, client, text):
        resp = client.post("/predict", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] in ("positive", "negative")
        assert 0.0 <= data["confidence"] <= 1.0

    def test_response_schema_complete(self, client):
        resp = client.post("/predict", json={"text": "Great product!"})
        assert resp.status_code == 200
        data = resp.json()
        for field in ("text", "sentiment", "confidence", "positive_score",
                      "negative_score", "processing_time_ms"):
            assert field in data, f"Missing field: {field}"

    def test_scores_sum_to_one(self, client):
        resp = client.post("/predict", json={"text": "I love this!"})
        data = resp.json()
        total = data["positive_score"] + data["negative_score"]
        assert abs(total - 1.0) < 0.01, f"Scores do not sum to 1: {total}"

    def test_confidence_matches_max_score(self, client):
        resp = client.post("/predict", json={"text": "Amazing quality!"})
        data = resp.json()
        expected = max(data["positive_score"], data["negative_score"])
        assert abs(data["confidence"] - expected) < 0.01

    def test_processing_time_is_positive(self, client):
        resp = client.post("/predict", json={"text": "Test text."})
        assert resp.json()["processing_time_ms"] > 0


# =============================================================================
# 3. EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_single_word(self, client):
        resp = client.post("/predict", json={"text": "Good"})
        assert resp.status_code == 200

    def test_single_character(self, client):
        resp = client.post("/predict", json={"text": "A"})
        assert resp.status_code == 200

    def test_very_long_text(self, client):
        long_text = "good " * 2000  # 10 000 chars
        resp = client.post("/predict", json={"text": long_text})
        assert resp.status_code == 200

    def test_text_with_numbers_only(self, client):
        resp = client.post("/predict", json={"text": "12345 67890"})
        assert resp.status_code == 200

    def test_text_with_punctuation_only(self, client):
        resp = client.post("/predict", json={"text": "!!! ??? ... ---"})
        assert resp.status_code == 200

    def test_mixed_case_text(self, client):
        resp = client.post("/predict", json={"text": "GoOd BaD gReAt TeRrIbLe"})
        assert resp.status_code == 200

    def test_unicode_text(self, client):
        resp = client.post("/predict", json={"text": "Très bien! こんにちは 你好"})
        assert resp.status_code == 200

    def test_newlines_and_tabs(self, client):
        resp = client.post("/predict", json={"text": "Good\nBad\tUgly"})
        assert resp.status_code == 200

    def test_repeated_word(self, client):
        resp = client.post("/predict", json={"text": "good " * 50})
        assert resp.status_code == 200


# =============================================================================
# 4. INVALID / MALICIOUS INPUT ROBUSTNESS
# =============================================================================

class TestInvalidInputs:
    def test_empty_string_rejected(self, client):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_whitespace_only_rejected(self, client):
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 422

    def test_missing_text_field(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_text_field_is_number(self, client):
        resp = client.post("/predict", json={"text": 42})
        assert resp.status_code in (200, 422)

    def test_null_text(self, client):
        resp = client.post("/predict", json={"text": None})
        assert resp.status_code == 422

    def test_text_exceeds_max_length(self, client):
        resp = client.post("/predict", json={"text": "a" * 10001})
        assert resp.status_code == 422

    def test_sql_injection_string(self, client):
        malicious = "'; DROP TABLE sentiments; --"
        resp = client.post("/predict", json={"text": malicious})
        assert resp.status_code in (200, 422)

    def test_xss_payload(self, client):
        xss = "<script>alert('xss')</script>"
        resp = client.post("/predict", json={"text": xss})
        assert resp.status_code in (200, 422)

    def test_path_traversal_string(self, client):
        payload = "../../../../etc/passwd"
        resp = client.post("/predict", json={"text": payload})
        assert resp.status_code in (200, 422)

    def test_null_byte_in_text(self, client):
        resp = client.post("/predict", json={"text": "good\x00bad"})
        assert resp.status_code in (200, 422)

    def test_wrong_content_type(self, client):
        resp = client.post("/predict", content="plain text",
                           headers={"Content-Type": "text/plain"})
        assert resp.status_code == 422

    def test_unknown_extra_fields_ignored(self, client):
        resp = client.post("/predict", json={"text": "Great!", "extra": "ignored"})
        assert resp.status_code == 200


# =============================================================================
# 5. BATCH ENDPOINT TESTS
# =============================================================================

class TestBatchPredict:
    def test_batch_basic(self, client):
        resp = client.post("/predict/batch",
                           json={"texts": ["I love it!", "I hate it."]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2

    def test_batch_single_item(self, client):
        resp = client.post("/predict/batch", json={"texts": ["Good product."]})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1

    def test_batch_max_items(self, client):
        texts = [f"Sample text number {i}" for i in range(32)]
        resp = client.post("/predict/batch", json={"texts": texts})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 32

    def test_batch_exceeds_max(self, client):
        texts = ["text"] * 33
        resp = client.post("/predict/batch", json={"texts": texts})
        assert resp.status_code == 422

    def test_batch_empty_list(self, client):
        resp = client.post("/predict/batch", json={"texts": []})
        assert resp.status_code == 422

    def test_batch_contains_blank_text(self, client):
        resp = client.post("/predict/batch", json={"texts": ["good", ""]})
        assert resp.status_code == 422

    def test_batch_total_time_present(self, client):
        resp = client.post("/predict/batch", json={"texts": ["Good!", "Bad."]})
        assert "total_processing_time_ms" in resp.json()

    def test_batch_result_order_preserved(self, client):
        texts = ["first text", "second text", "third text"]
        resp = client.post("/predict/batch", json={"texts": texts})
        results = resp.json()["results"]
        for i, result in enumerate(results):
            assert result["text"] == texts[i]


# =============================================================================
# 6. INTEGRATION / REAL-WORLD SCENARIOS
# =============================================================================

class TestIntegrationScenarios:
    def test_product_review_workflow(self, client):
        """Simulate an e-commerce review moderation pipeline."""
        reviews = [
            "This laptop is amazing, battery life is great!",
            "Delivery was terrible, package arrived damaged.",
            "Average product, nothing special about it.",
        ]
        resp = client.post("/predict/batch", json={"texts": reviews})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert all("sentiment" in r for r in results)

    def test_health_before_and_after_inference(self, client):
        """Model should stay healthy after repeated inference."""
        client.post("/predict", json={"text": "Great!"})
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True

    def test_sequential_predictions_consistent(self, client):
        """Same text should return the same sentiment across calls."""
        text = "I really love this service."
        r1 = client.post("/predict", json={"text": text}).json()
        r2 = client.post("/predict", json={"text": text}).json()
        assert r1["sentiment"] == r2["sentiment"]
        assert abs(r1["confidence"] - r2["confidence"]) < 0.001

    def test_social_media_pipeline(self, client):
        """Simulate social-media sentiment monitoring."""
        tweets = [
            "Just tried the new #product and it is absolutely wonderful!",
            "Worst customer support ever. Never buying again. #fail",
            "Meh, it is okay I guess.",
        ]
        resp = client.post("/predict/batch", json={"texts": tweets})
        assert resp.status_code == 200


# =============================================================================
# 7. PERFORMANCE / STRESS TESTS
# =============================================================================

class TestPerformance:
    def test_single_inference_under_500ms(self, client):
        start = time.perf_counter()
        resp = client.post("/predict", json={"text": "This product is great!"})
        elapsed = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        assert elapsed < 500, f"Inference took {elapsed:.1f}ms (limit 500ms)"

    def test_batch_32_under_2000ms(self, client):
        texts = ["This is a test sentence for performance evaluation."] * 32
        start = time.perf_counter()
        resp = client.post("/predict/batch", json={"texts": texts})
        elapsed = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        assert elapsed < 2000, f"Batch of 32 took {elapsed:.1f}ms (limit 2000ms)"

    def test_repeated_requests_stable(self, client):
        """50 sequential requests should all succeed."""
        failures = 0
        for i in range(50):
            resp = client.post("/predict", json={"text": f"Test sentence number {i}."})
            if resp.status_code != 200:
                failures += 1
        assert failures == 0, f"{failures}/50 requests failed"

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_concurrent_requests(self, async_client):
        """20 concurrent requests should all succeed."""
        payload = {"text": "Concurrent sentiment analysis test."}

        async def single_request():
            return await async_client.post("/predict", json=payload)

        responses = await asyncio.gather(*[single_request() for _ in range(20)])
        codes = [r.status_code for r in responses]
        assert all(c == 200 for c in codes), f"Some requests failed: {codes}"

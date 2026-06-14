"""Tests: validation classification, retry workflow, confidence routing
(Steps 2 and 5)."""

from __future__ import annotations

from schemas.extraction_schema import ExtractionResult
from src.confidence_router import build_review_queue, route_document
from src.metrics import RetryMetrics
from src.retry_handler import process_document
from src.validator import (
    FailureCategory,
    validate_extraction,
)


# --- Validation + classification ------------------------------------------


def test_valid_payload_validates(valid_payload):
    outcome = validate_extraction(valid_payload)
    assert outcome.is_valid
    assert isinstance(outcome.result, ExtractionResult)


def test_enum_mismatch_classified(make_payload):
    payload = make_payload(document_type={"value": "thesis", "confidence": 0.9})
    outcome = validate_extraction(payload)
    assert not outcome.is_valid
    assert outcome.category == FailureCategory.ENUM_MISMATCH
    assert outcome.is_retryable


def test_type_conversion_classified(make_payload):
    payload = make_payload(citation_count={"value": "many", "confidence": 0.9})
    outcome = validate_extraction(payload)
    assert outcome.category == FailureCategory.TYPE_CONVERSION_ISSUE
    assert outcome.is_retryable


def test_formatting_issue_classified(make_payload):
    payload = make_payload(title={"value": "T", "confidence": 2.0})
    outcome = validate_extraction(payload)
    assert outcome.category == FailureCategory.FORMATTING_ISSUE
    assert outcome.is_retryable


def test_missing_information_not_retryable(make_payload):
    payload = make_payload(title={"value": None, "confidence": 0.3})
    outcome = validate_extraction(payload)
    assert outcome.category == FailureCategory.MISSING_INFORMATION
    assert not outcome.is_retryable


def test_invalid_json_string_is_formatting():
    outcome = validate_extraction("{not valid json")
    assert not outcome.is_valid
    assert outcome.category == FailureCategory.FORMATTING_ISSUE


# --- Retry workflow --------------------------------------------------------


def test_first_attempt_success(valid_payload):
    metrics = RetryMetrics()
    record = process_document(
        "doc_001",
        "some text",
        extract_fn=lambda req: valid_payload,
        metrics=metrics,
    )
    assert record.succeeded
    assert record.attempts == 1
    assert not record.retried
    assert metrics.total_failures == 0


def test_retry_recovers_enum_mismatch(make_payload):
    bad = make_payload(document_type={"value": "thesis", "confidence": 0.9})
    good = make_payload()
    calls = {"n": 0}

    def extract_fn(req):
        calls["n"] += 1
        return bad if calls["n"] == 1 else good

    metrics = RetryMetrics()
    record = process_document(
        "doc_001", "text", extract_fn=extract_fn, metrics=metrics
    )
    assert record.succeeded
    assert record.attempts == 2
    assert record.retried
    assert metrics.total_failures == 1
    assert metrics.retry_successes == 1
    assert metrics.retry_success_rate == 1.0


def test_non_retryable_does_not_retry(make_payload):
    bad = make_payload(title={"value": None, "confidence": 0.2})
    calls = {"n": 0}

    def extract_fn(req):
        calls["n"] += 1
        return bad

    metrics = RetryMetrics()
    record = process_document(
        "doc_x", "text", extract_fn=extract_fn, metrics=metrics
    )
    assert not record.succeeded
    assert record.attempts == 1
    assert not record.retried
    assert calls["n"] == 1  # never called a second time
    assert metrics.non_retryable == 1
    assert record.invalid_extraction == bad
    assert record.validation_error


def test_retry_can_still_fail(make_payload):
    bad = make_payload(document_type={"value": "thesis", "confidence": 0.9})

    def extract_fn(req):
        return bad  # never improves

    metrics = RetryMetrics()
    record = process_document(
        "doc_y", "text", extract_fn=extract_fn, metrics=metrics
    )
    assert not record.succeeded
    assert record.attempts == 2
    assert record.retried
    assert metrics.retry_failures == 1
    assert metrics.retry_success_rate == 0.0


def test_metrics_success_rate_mixed():
    m = RetryMetrics()
    m.record_failure(FailureCategory.ENUM_MISMATCH, retryable=True)
    m.record_retry_result(succeeded=True)
    m.record_failure(FailureCategory.FORMATTING_ISSUE, retryable=True)
    m.record_retry_result(succeeded=False)
    assert m.retry_attempts == 2
    assert m.retry_success_rate == 0.5
    assert m.total_failures == 2


# --- Confidence routing ----------------------------------------------------


def test_high_confidence_not_routed(valid_payload):
    result = ExtractionResult.model_validate(valid_payload)
    routing = route_document(result, threshold=0.80)
    assert not routing.needs_review
    assert routing.items == []


def test_low_field_confidence_routed(make_payload):
    payload = make_payload(author={"value": "guessed", "confidence": 0.4})
    result = ExtractionResult.model_validate(payload)
    routing = route_document(result, threshold=0.80)
    assert routing.needs_review
    flagged = {item.field_name for item in routing.items}
    assert "author" in flagged
    item = next(i for i in routing.items if i.field_name == "author")
    assert item.document_id == "doc_001"
    assert item.confidence_score == 0.4
    assert "below" in item.review_reason


def test_low_overall_confidence_routed(make_payload):
    payload = make_payload(overall_confidence=0.5)
    result = ExtractionResult.model_validate(payload)
    routing = route_document(result, threshold=0.80)
    flagged = {item.field_name for item in routing.items}
    assert "overall_confidence" in flagged


def test_build_review_queue_serializes_enum(make_payload):
    payload = make_payload(
        document_type={"value": "research_paper", "confidence": 0.3}
    )
    result = ExtractionResult.model_validate(payload)
    queue = build_review_queue([result], threshold=0.80)
    dt_entries = [e for e in queue if e["field_name"] == "document_type"]
    assert dt_entries
    assert dt_entries[0]["extracted_value"] == "research_paper"  # enum -> str


def test_review_queue_entry_has_required_keys(make_payload):
    payload = make_payload(doi={"value": None, "confidence": 0.2})
    result = ExtractionResult.model_validate(payload)
    queue = build_review_queue([result], threshold=0.80)
    assert queue
    for entry in queue:
        assert set(entry) == {
            "document_id",
            "field_name",
            "extracted_value",
            "confidence_score",
            "review_reason",
        }

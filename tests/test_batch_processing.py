"""Tests: batch processing workflow and chunking strategy (Step 4)."""

from __future__ import annotations

from src.batch_processor import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    BatchItem,
    build_batch_requests,
    chunk_document,
    find_failures,
    recombine_chunk_extractions,
    simulate_batch,
)


# --- Batch request construction -------------------------------------------


def test_build_batch_requests_tags_custom_ids():
    items = [BatchItem(custom_id=f"doc_{i:03d}", document_text="x") for i in range(3)]
    requests = build_batch_requests(items)
    assert [r["custom_id"] for r in requests] == ["doc_000", "doc_001", "doc_002"]
    # Each request forces the extraction tool.
    for r in requests:
        assert r["params"]["tool_choice"]["name"] == "extract_document"
        assert r["params"]["model"] == "claude-opus-4-8"


# --- Simulated 100-document workflow --------------------------------------


def test_simulate_100_documents_all_succeed():
    items = [BatchItem(custom_id=f"doc_{i:03d}", document_text="x") for i in range(100)]
    report = simulate_batch(items)
    assert report.submitted == 100
    assert len(report.succeeded) == 100
    assert report.failed == []


def test_simulate_with_forced_failures_and_recovery():
    items = [BatchItem(custom_id=f"doc_{i:03d}", document_text="x") for i in range(100)]
    failures = {"doc_017", "doc_048", "doc_081"}
    report = simulate_batch(items, forced_failures=failures)
    assert set(report.failed) == failures
    assert set(report.resubmitted) == failures
    assert set(report.recovered) == failures
    summary = report.to_dict()
    assert summary["final_success_count"] == 100
    assert summary["failed_count"] == 3


def test_simulate_failures_without_recovery():
    items = [BatchItem(custom_id=f"doc_{i:03d}", document_text="x") for i in range(10)]
    report = simulate_batch(
        items, forced_failures={"doc_003"}, recover_on_resubmit=False
    )
    assert report.recovered == []
    assert report.to_dict()["final_success_count"] == 9


def test_find_failures_identifies_non_succeeded():
    results = {
        "doc_000": {"status": "succeeded", "extraction": {}},
        "doc_001": {"status": "errored", "error": "invalid_request"},
        "doc_002": {"status": "expired"},
    }
    assert set(find_failures(results)) == {"doc_001", "doc_002"}


# --- Chunking --------------------------------------------------------------


def test_short_document_single_chunk():
    chunks = chunk_document("short text")
    assert chunks == ["short text"]


def test_long_document_splits_with_overlap():
    text = "A" * 20000
    chunks = chunk_document(text, chunk_size=8000, overlap=800)
    assert len(chunks) > 1
    assert all(len(c) <= 8000 for c in chunks)
    # Reassembling with the documented overlap reproduces the original length.
    step = 8000 - 800
    assert chunks[0][-800:] == chunks[1][:800] or text[step : step + 800] == chunks[1][:800]


def test_chunk_overlap_must_be_smaller_than_size():
    try:
        chunk_document("x" * 100, chunk_size=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError for overlap >= chunk_size")


def test_chunk_defaults_documented():
    assert CHUNK_SIZE_CHARS == 8000
    assert CHUNK_OVERLAP_CHARS == 800


# --- Recombination ---------------------------------------------------------


def test_recombine_single_extraction_passthrough():
    one = {"title": {"value": "T", "confidence": 0.9}, "overall_confidence": 0.9}
    assert recombine_chunk_extractions([one]) == one


def test_recombine_keeps_highest_confidence_per_field():
    chunk_a = {
        "title": {"value": "From A", "confidence": 0.6},
        "doi": {"value": None, "confidence": 0.5},
        "overall_confidence": 0.6,
    }
    chunk_b = {
        "title": {"value": "From B", "confidence": 0.9},
        "doi": {"value": "10.1/x", "confidence": 0.95},
        "overall_confidence": 0.8,
    }
    combined = recombine_chunk_extractions([chunk_a, chunk_b])
    assert combined["title"]["value"] == "From B"
    assert combined["doi"]["value"] == "10.1/x"
    # overall_confidence averages the chunks.
    assert abs(combined["overall_confidence"] - 0.7) < 1e-9


def test_recombine_empty_list():
    assert recombine_chunk_extractions([]) == {}

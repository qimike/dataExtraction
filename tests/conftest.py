"""Shared test fixtures and helpers.

Adds the repo root to ``sys.path`` so ``schemas`` / ``src`` import cleanly when
pytest is invoked from anywhere, and provides a builder for a valid extraction
payload that individual tests mutate to exercise specific failure modes.
"""

from __future__ import annotations

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _valid_payload() -> dict:
    """A complete, schema-valid extraction payload."""
    return {
        "document_id": {"value": "doc_001", "confidence": 1.0},
        "title": {"value": "A Valid Title", "confidence": 0.95},
        "document_type": {"value": "research_paper", "confidence": 0.93},
        "summary": {"value": "A short, valid summary of the document.", "confidence": 0.9},
        "other_document_type_detail": None,
        "publication_date": {"value": "2024-01-01", "confidence": 0.8},
        "author": {"value": "A. Author", "confidence": 0.9},
        "organization": {"value": "Example Org", "confidence": 0.85},
        "email": {"value": "a.author@example.org", "confidence": 0.97},
        "phone_number": {"value": None, "confidence": 0.98},
        "doi": {"value": "10.1000/xyz123", "confidence": 0.95},
        "citation_count": {"value": 42, "confidence": 0.9},
        "overall_confidence": 0.9,
    }


@pytest.fixture
def valid_payload() -> dict:
    return _valid_payload()


@pytest.fixture
def make_payload():
    """Return a factory that yields fresh deep copies of the valid payload."""

    def _factory(**overrides) -> dict:
        payload = copy.deepcopy(_valid_payload())
        payload.update(overrides)
        return payload

    return _factory

"""Tests: schema validation, enum validation, nullable fields (Step 1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.extraction_schema import (
    ConfidenceField,
    DocumentType,
    ExtractionResult,
    extraction_tool_schema,
)


# --- Schema validation -----------------------------------------------------


def test_valid_payload_parses(valid_payload):
    result = ExtractionResult.model_validate(valid_payload)
    assert result.document_id.value == "doc_001"
    assert result.document_type.value == DocumentType.RESEARCH_PAPER
    assert result.overall_confidence == 0.9


def test_required_field_null_value_rejected(make_payload):
    payload = make_payload(title={"value": None, "confidence": 0.5})
    with pytest.raises(ValidationError) as exc:
        ExtractionResult.model_validate(payload)
    assert "title" in str(exc.value)


def test_required_field_empty_string_rejected(make_payload):
    payload = make_payload(summary={"value": "   ", "confidence": 0.5})
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(payload)


def test_confidence_out_of_range_rejected(make_payload):
    payload = make_payload(title={"value": "T", "confidence": 1.4})
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(payload)


def test_overall_confidence_bounds(make_payload):
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(make_payload(overall_confidence=-0.1))
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(make_payload(overall_confidence=1.1))


def test_unknown_field_rejected(make_payload):
    payload = make_payload(surprise={"value": "x", "confidence": 0.5})
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(payload)


def test_tool_schema_is_a_dict_with_properties():
    schema = extraction_tool_schema()
    assert schema["type"] == "object"
    assert "document_id" in schema["properties"]
    assert "overall_confidence" in schema["properties"]


# --- Enum validation -------------------------------------------------------


def test_enum_accepts_allowed_values(make_payload):
    for value in [t.value for t in DocumentType if t != DocumentType.OTHER]:
        payload = make_payload(document_type={"value": value, "confidence": 0.9})
        result = ExtractionResult.model_validate(payload)
        assert result.document_type.value.value == value


def test_enum_rejects_unknown_value(make_payload):
    payload = make_payload(document_type={"value": "thesis", "confidence": 0.9})
    with pytest.raises(ValidationError) as exc:
        ExtractionResult.model_validate(payload)
    assert any(e["type"] == "enum" for e in exc.value.errors())


def test_other_requires_detail(make_payload):
    payload = make_payload(
        document_type={"value": "other", "confidence": 0.9},
        other_document_type_detail=None,
    )
    with pytest.raises(ValidationError) as exc:
        ExtractionResult.model_validate(payload)
    assert "other_document_type_detail" in str(exc.value)


def test_other_with_detail_is_valid(make_payload):
    payload = make_payload(
        document_type={"value": "other", "confidence": 0.9},
        other_document_type_detail={"value": "technical specification", "confidence": 0.8},
    )
    result = ExtractionResult.model_validate(payload)
    assert result.other_document_type_detail.value == "technical specification"


def test_detail_must_be_null_for_non_other(make_payload):
    payload = make_payload(
        document_type={"value": "report", "confidence": 0.9},
        other_document_type_detail={"value": "something", "confidence": 0.8},
    )
    with pytest.raises(ValidationError) as exc:
        ExtractionResult.model_validate(payload)
    assert "must be null" in str(exc.value)


# --- Nullable fields -------------------------------------------------------


def test_nullable_fields_accept_null(make_payload):
    payload = make_payload(
        email={"value": None, "confidence": 0.99},
        phone_number={"value": None, "confidence": 0.99},
        doi={"value": None, "confidence": 0.99},
        citation_count={"value": None, "confidence": 0.99},
    )
    result = ExtractionResult.model_validate(payload)
    assert result.email.value is None
    assert result.doi.value is None
    assert result.citation_count.value is None


def test_nullable_fields_accept_values(make_payload):
    result = ExtractionResult.model_validate(make_payload())
    assert result.email.value == "a.author@example.org"
    assert result.citation_count.value == 42


def test_optional_fields_can_be_omitted(valid_payload):
    valid_payload.pop("publication_date")
    valid_payload.pop("author")
    valid_payload.pop("organization")
    result = ExtractionResult.model_validate(valid_payload)
    assert result.publication_date is None
    assert result.author is None


def test_confidence_field_value_optional_by_default():
    field = ConfidenceField[str](confidence=0.5)
    assert field.value is None

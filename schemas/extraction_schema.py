"""Structured extraction schema (Step 1).

Defines the Pydantic models that constrain what Claude returns when extracting
metadata from a document. The schema encodes three classes of field
(required / optional / nullable), a closed ``DocumentType`` enum with a
conditional detail field, and a per-field confidence wrapper.

The same models serve three purposes:

1. Generate the JSON Schema handed to Claude as a tool ``input_schema``
   (see ``src/extractor.py``) so the model is forced into the right shape.
2. Validate every extraction the model returns (see ``src/validator.py``).
3. Document the contract for human reviewers.

Design notes
------------
* Every field is wrapped in :class:`ConfidenceField` so the model reports a
  ``value`` and a ``confidence`` (0.0-1.0) for each one.
* **Required** fields (``document_id``, ``title``, ``document_type``,
  ``summary``) must have a non-null ``value`` — enforced in
  :meth:`ExtractionResult.check_required_values`.
* **Optional** fields (``publication_date``, ``author``, ``organization``)
  are typed ``Optional[ConfidenceField]`` — the whole wrapper may be omitted.
* **Nullable** fields (``email``, ``phone_number``, ``doi``,
  ``citation_count``) always carry a wrapper, but ``value`` may be ``None``.
  The model is instructed to return ``null`` rather than fabricate a value.
"""

from __future__ import annotations

from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


class DocumentType(str, Enum):
    """Closed set of document categories.

    ``OTHER`` is an escape hatch that *requires* a free-text
    ``other_document_type_detail`` so the category is never silently lost.
    """

    RESEARCH_PAPER = "research_paper"
    REPORT = "report"
    ARTICLE = "article"
    INVOICE = "invoice"
    CONTRACT = "contract"
    OTHER = "other"


class ConfidenceField(BaseModel, Generic[T]):
    """A single extracted field carrying a value and a confidence score.

    ``value`` is ``Optional`` so the same wrapper works for nullable fields
    (the model returns ``value=None`` when the information is absent).
    ``confidence`` is constrained to the inclusive range ``[0.0, 1.0]``.
    """

    model_config = ConfigDict(extra="forbid")

    value: Optional[T] = Field(
        default=None,
        description="Extracted value, or null when the information is absent.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in this field, from 0.0 to 1.0.",
    )


class ExtractionResult(BaseModel):
    """Full structured extraction for one document."""

    model_config = ConfigDict(extra="forbid")

    # --- Required fields (value must be non-null) ---------------------------
    document_id: ConfidenceField[str] = Field(
        ..., description="Stable identifier for the document (e.g. doc_001)."
    )
    title: ConfidenceField[str] = Field(..., description="Document title.")
    document_type: ConfidenceField[DocumentType] = Field(
        ..., description="One of the allowed DocumentType values."
    )
    summary: ConfidenceField[str] = Field(
        ..., description="Concise 1-3 sentence summary of the document."
    )

    # --- Conditional detail for document_type == other ----------------------
    other_document_type_detail: Optional[ConfidenceField[str]] = Field(
        default=None,
        description=(
            "Required when document_type == 'other'; must be null otherwise."
        ),
    )

    # --- Optional fields (wrapper may be omitted entirely) ------------------
    publication_date: Optional[ConfidenceField[str]] = Field(
        default=None, description="Publication date as an ISO-8601 string."
    )
    author: Optional[ConfidenceField[str]] = Field(
        default=None, description="Primary author or authors."
    )
    organization: Optional[ConfidenceField[str]] = Field(
        default=None, description="Publishing or issuing organization."
    )

    # --- Nullable fields (wrapper present, value may be null) ---------------
    email: ConfidenceField[str] = Field(
        ..., description="Contact email, or null if none present."
    )
    phone_number: ConfidenceField[str] = Field(
        ..., description="Contact phone number, or null if none present."
    )
    doi: ConfidenceField[str] = Field(
        ..., description="Digital Object Identifier, or null if none present."
    )
    citation_count: ConfidenceField[int] = Field(
        ..., description="Citation count, or null if not stated."
    )

    # --- Document-level confidence ------------------------------------------
    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence across the whole extraction.",
    )

    @model_validator(mode="after")
    def check_required_values(self) -> "ExtractionResult":
        """Required fields must carry a non-null, non-empty ``value``."""
        required = {
            "document_id": self.document_id,
            "title": self.title,
            "document_type": self.document_type,
            "summary": self.summary,
        }
        for name, field in required.items():
            if field.value is None:
                raise ValueError(f"Required field '{name}' must have a non-null value.")
            if isinstance(field.value, str) and not field.value.strip():
                raise ValueError(f"Required field '{name}' must not be empty.")
        return self

    @model_validator(mode="after")
    def check_other_detail(self) -> "ExtractionResult":
        """Enforce the ``other`` <-> ``other_document_type_detail`` contract.

        * detail is **required** (non-null value) when type == ``other``
        * detail must be **null** for any other type
        """
        is_other = self.document_type.value == DocumentType.OTHER
        detail = self.other_document_type_detail

        if is_other:
            if detail is None or detail.value is None or not str(detail.value).strip():
                raise ValueError(
                    "other_document_type_detail is required when "
                    "document_type == 'other'."
                )
        else:
            if detail is not None and detail.value is not None:
                raise ValueError(
                    "other_document_type_detail must be null unless "
                    "document_type == 'other'."
                )
        return self


def extraction_tool_schema() -> dict:
    """Return the JSON Schema used as the Claude tool ``input_schema``.

    Pydantic generates a JSON-Schema-compatible dict directly from the model,
    keeping the tool contract and the validation contract in lockstep.
    """
    return ExtractionResult.model_json_schema()


__all__ = [
    "DocumentType",
    "ConfidenceField",
    "ExtractionResult",
    "extraction_tool_schema",
]

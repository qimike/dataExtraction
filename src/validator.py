"""Validation layer (Step 2, part 1).

Wraps Pydantic validation of :class:`ExtractionResult` and classifies any
failure into a retryable or non-retryable category. The classification drives
the retry handler: retryable failures are worth re-prompting the model with the
error attached; non-retryable failures reflect missing/ambiguous source data
and should go straight to human review instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import ValidationError

# Allow running both as ``python -m src.validator`` and via package import.
try:
    from schemas.extraction_schema import ExtractionResult
except ModuleNotFoundError:  # pragma: no cover - path shim for ad-hoc runs
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from schemas.extraction_schema import ExtractionResult


class FailureCategory(str, Enum):
    """Why a validation attempt failed.

    The first three are **retryable** — the document almost certainly contains
    the information, the model just returned it in the wrong shape. The last
    three are **non-retryable** — re-prompting will not conjure data that is not
    in the source, so these route to human review.
    """

    # Retryable -------------------------------------------------------------
    FORMATTING_ISSUE = "formatting_issue"
    ENUM_MISMATCH = "enum_mismatch"
    TYPE_CONVERSION_ISSUE = "type_conversion_issue"
    # Non-retryable ---------------------------------------------------------
    MISSING_INFORMATION = "missing_information"
    AMBIGUOUS_INFORMATION = "ambiguous_information"
    DOCUMENT_CORRUPTION = "document_corruption"


RETRYABLE_CATEGORIES = frozenset(
    {
        FailureCategory.FORMATTING_ISSUE,
        FailureCategory.ENUM_MISMATCH,
        FailureCategory.TYPE_CONVERSION_ISSUE,
    }
)


@dataclass
class ValidationOutcome:
    """Result of validating a raw extraction payload."""

    is_valid: bool
    result: Optional[ExtractionResult] = None
    category: Optional[FailureCategory] = None
    error_message: str = ""
    raw_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_retryable(self) -> bool:
        return self.category in RETRYABLE_CATEGORIES


def classify_validation_error(exc: ValidationError) -> FailureCategory:
    """Map a Pydantic ``ValidationError`` to a :class:`FailureCategory`.

    The heuristics inspect the structured error list rather than string
    matching the rendered message, so they are stable across Pydantic
    versions.
    """
    errors = exc.errors()

    def has(pred) -> bool:
        return any(pred(e) for e in errors)

    # An invalid enum literal -> the model chose a value outside DocumentType.
    if has(lambda e: e["type"] == "enum"):
        return FailureCategory.ENUM_MISMATCH

    # Wrong primitive type (e.g. citation_count returned as "many").
    type_codes = {
        "int_parsing",
        "int_type",
        "float_parsing",
        "float_type",
        "bool_parsing",
        "string_type",
    }
    if has(lambda e: e["type"] in type_codes):
        return FailureCategory.TYPE_CONVERSION_ISSUE

    # A required field missing entirely usually means the model could not find
    # it in the source -> route to a human rather than retrying forever.
    if has(lambda e: e["type"] == "missing"):
        return FailureCategory.MISSING_INFORMATION

    # A required field present but null/empty (our custom validator) -> the
    # information is absent or ambiguous in the document.
    if has(lambda e: "must have a non-null value" in str(e.get("msg", ""))):
        return FailureCategory.MISSING_INFORMATION

    # Confidence out of range, extra keys, malformed structure -> formatting.
    return FailureCategory.FORMATTING_ISSUE


def validate_extraction(payload: dict[str, Any] | str) -> ValidationOutcome:
    """Validate a raw extraction payload against :class:`ExtractionResult`.

    Parameters
    ----------
    payload:
        Either a ``dict`` (already-parsed tool input) or a JSON ``str``.

    Returns
    -------
    ValidationOutcome
        ``is_valid`` plus, on failure, the classified category and a
        human-readable error message suitable for embedding in a retry prompt.
    """
    # JSON-decode first so malformed JSON is itself a formatting failure.
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            return ValidationOutcome(
                is_valid=False,
                category=FailureCategory.DOCUMENT_CORRUPTION
                if not payload.strip()
                else FailureCategory.FORMATTING_ISSUE,
                error_message=f"Payload is not valid JSON: {exc}",
            )

    try:
        result = ExtractionResult.model_validate(payload)
    except ValidationError as exc:
        return ValidationOutcome(
            is_valid=False,
            category=classify_validation_error(exc),
            error_message=_render_errors(exc),
            raw_errors=exc.errors(),
        )

    return ValidationOutcome(is_valid=True, result=result)


def _render_errors(exc: ValidationError) -> str:
    """Render a compact, model-friendly description of each error."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"- field '{loc}': {err['msg']} (type={err['type']})")
    return "Validation failed:\n" + "\n".join(lines)


__all__ = [
    "FailureCategory",
    "RETRYABLE_CATEGORIES",
    "ValidationOutcome",
    "classify_validation_error",
    "validate_extraction",
]

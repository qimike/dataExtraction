"""Validation-retry loop (Step 2, part 2).

Orchestrates a single document through extract -> validate -> (retry once) and
records the outcome in :class:`RetryMetrics`. The policy is deliberately
conservative:

1. Extract, then validate.
2. If valid, done.
3. If invalid and the failure is **retryable** (formatting / enum / type), build
   a retry request carrying the original document, the previous extraction, and
   the validation error, then extract once more and re-validate.
4. If invalid and **non-retryable** (missing / ambiguous / corruption), do not
   retry — the data is not in the source. Surface it for human review.

Retrying exactly once bounds cost and latency while still recovering the large
majority of shape-only failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from schemas.extraction_schema import ExtractionResult
from src.extractor import ExtractionRequest, extract
from src.metrics import RetryMetrics
from src.validator import FailureCategory, validate_extraction

# An extractor function: takes a request, returns the raw extraction dict.
ExtractFn = Callable[[ExtractionRequest], dict[str, Any]]


@dataclass
class RetryRecord:
    """Full audit trail for one document's journey through the loop."""

    document_id: str
    succeeded: bool
    attempts: int
    result: Optional[ExtractionResult] = None
    final_category: Optional[FailureCategory] = None
    retried: bool = False
    # Captured context for non-retryable / failed-retry cases:
    source_document: str = ""
    invalid_extraction: Optional[dict[str, Any]] = None
    validation_error: str = ""

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "succeeded": self.succeeded,
            "attempts": self.attempts,
            "retried": self.retried,
            "final_category": self.final_category.value
            if self.final_category
            else None,
            "invalid_extraction": self.invalid_extraction,
            "validation_error": self.validation_error,
        }


def process_document(
    document_id: str,
    document_text: str,
    *,
    extract_fn: Optional[ExtractFn] = None,
    metrics: Optional[RetryMetrics] = None,
    include_few_shot: bool = True,
) -> RetryRecord:
    """Run one document through the extract/validate/retry loop.

    Parameters
    ----------
    extract_fn:
        Injectable extractor. Defaults to the live Claude extractor; tests pass
        a deterministic stub. It receives an :class:`ExtractionRequest` and
        returns the raw extraction ``dict``.
    metrics:
        Optional :class:`RetryMetrics` accumulator updated in place.
    """
    if extract_fn is None:
        extract_fn = lambda req: extract(req)  # noqa: E731 - thin default
    if metrics is None:
        metrics = RetryMetrics()

    # --- Attempt 1 ---------------------------------------------------------
    first_req = ExtractionRequest(
        document_id=document_id,
        document_text=document_text,
        include_few_shot=include_few_shot,
    )
    raw = extract_fn(first_req)
    outcome = validate_extraction(raw)

    if outcome.is_valid:
        return RetryRecord(
            document_id=document_id,
            succeeded=True,
            attempts=1,
            result=outcome.result,
        )

    # Record the first failure and decide whether to retry.
    metrics.record_failure(outcome.category, retryable=outcome.is_retryable)

    if not outcome.is_retryable:
        # Non-retryable: capture context and stop. Goes to human review.
        return RetryRecord(
            document_id=document_id,
            succeeded=False,
            attempts=1,
            final_category=outcome.category,
            retried=False,
            source_document=document_text,
            invalid_extraction=raw,
            validation_error=outcome.error_message,
        )

    # --- Attempt 2 (retry once) -------------------------------------------
    retry_req = ExtractionRequest(
        document_id=document_id,
        document_text=document_text,
        include_few_shot=include_few_shot,
        previous_extraction=raw,
        validation_error=outcome.error_message,
    )
    retry_raw = extract_fn(retry_req)
    retry_outcome = validate_extraction(retry_raw)

    metrics.record_retry_result(succeeded=retry_outcome.is_valid)

    if retry_outcome.is_valid:
        return RetryRecord(
            document_id=document_id,
            succeeded=True,
            attempts=2,
            retried=True,
            result=retry_outcome.result,
        )

    return RetryRecord(
        document_id=document_id,
        succeeded=False,
        attempts=2,
        retried=True,
        final_category=retry_outcome.category,
        source_document=document_text,
        invalid_extraction=retry_raw,
        validation_error=retry_outcome.error_message,
    )


__all__ = ["RetryRecord", "process_document", "ExtractFn"]

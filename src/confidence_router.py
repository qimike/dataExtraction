"""Human-review routing (Step 5).

Any field whose confidence falls below a threshold (default 0.80), and any
document whose ``overall_confidence`` falls below it, is routed to a review
queue. The queue entries carry enough context for a human to act without
re-opening the source: document id, field name, extracted value, the
confidence score, and a human-readable reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from schemas.extraction_schema import ConfidenceField, ExtractionResult

DEFAULT_THRESHOLD = 0.80


@dataclass
class ReviewItem:
    """One flagged field destined for the human review queue."""

    document_id: str
    field_name: str
    extracted_value: Any
    confidence_score: float
    review_reason: str

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "field_name": self.field_name,
            "extracted_value": self.extracted_value,
            "confidence_score": self.confidence_score,
            "review_reason": self.review_reason,
        }


@dataclass
class RoutingResult:
    """Outcome of routing a single document."""

    document_id: str
    items: list[ReviewItem] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.items)

    @property
    def reasons(self) -> list[str]:
        return [item.review_reason for item in self.items]


def _iter_confidence_fields(result: ExtractionResult):
    """Yield ``(field_name, ConfidenceField)`` for every present wrapper."""
    for name, value in result.__dict__.items():
        if isinstance(value, ConfidenceField):
            yield name, value


def route_document(
    result: ExtractionResult, threshold: float = DEFAULT_THRESHOLD
) -> RoutingResult:
    """Route one extraction, flagging low-confidence fields and documents.

    A field is flagged when ``confidence < threshold``. The document is
    additionally flagged when ``overall_confidence < threshold`` even if no
    individual field tripped — a low aggregate signals systemic uncertainty.
    """
    doc_id = result.document_id.value or "<unknown>"
    routing = RoutingResult(document_id=doc_id)

    for name, conf_field in _iter_confidence_fields(result):
        if conf_field.confidence < threshold:
            routing.items.append(
                ReviewItem(
                    document_id=doc_id,
                    field_name=name,
                    extracted_value=_serialize(conf_field.value),
                    confidence_score=round(conf_field.confidence, 4),
                    review_reason=(
                        f"Field confidence {conf_field.confidence:.2f} is below "
                        f"the {threshold:.2f} threshold."
                    ),
                )
            )

    if result.overall_confidence < threshold:
        routing.items.append(
            ReviewItem(
                document_id=doc_id,
                field_name="overall_confidence",
                extracted_value=round(result.overall_confidence, 4),
                confidence_score=round(result.overall_confidence, 4),
                review_reason=(
                    f"Document overall confidence {result.overall_confidence:.2f} "
                    f"is below the {threshold:.2f} threshold."
                ),
            )
        )

    return routing


def build_review_queue(
    results: list[ExtractionResult], threshold: float = DEFAULT_THRESHOLD
) -> list[dict]:
    """Build the flat ``review_queue.json`` payload across many documents."""
    queue: list[dict] = []
    for result in results:
        routing = route_document(result, threshold)
        queue.extend(item.to_dict() for item in routing.items)
    return queue


def _serialize(value: Any) -> Any:
    """Make enum values JSON-friendly for the queue file."""
    if hasattr(value, "value"):
        return value.value
    return value


__all__ = [
    "DEFAULT_THRESHOLD",
    "ReviewItem",
    "RoutingResult",
    "route_document",
    "build_review_queue",
]

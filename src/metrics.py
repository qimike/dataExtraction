"""Metrics aggregation (Steps 2, 4, 5 reporting).

Pure, dependency-free accumulators so they can be unit-tested without touching
the Anthropic API. Three trackers:

* :class:`RetryMetrics`     — retry success/failure rates (Step 2)
* :class:`ReviewMetrics`    — human-review queue volume (Step 5)
* :class:`SLACalculator`    — sequential vs batch timing and SLA compliance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.validator import FailureCategory


@dataclass
class RetryMetrics:
    """Counts the outcomes of the validation-retry loop."""

    total_failures: int = 0
    retry_successes: int = 0
    retry_failures: int = 0
    non_retryable: int = 0
    by_category: dict[str, int] = field(default_factory=dict)

    def record_failure(self, category: FailureCategory, *, retryable: bool) -> None:
        self.total_failures += 1
        self.by_category[category.value] = self.by_category.get(category.value, 0) + 1
        if not retryable:
            self.non_retryable += 1

    def record_retry_result(self, *, succeeded: bool) -> None:
        if succeeded:
            self.retry_successes += 1
        else:
            self.retry_failures += 1

    @property
    def retry_attempts(self) -> int:
        return self.retry_successes + self.retry_failures

    @property
    def retry_success_rate(self) -> float:
        """Fraction of *attempted* retries that succeeded (0.0 if none)."""
        if self.retry_attempts == 0:
            return 0.0
        return self.retry_successes / self.retry_attempts

    def to_dict(self) -> dict:
        return {
            "total_failures": self.total_failures,
            "retry_successes": self.retry_successes,
            "retry_failures": self.retry_failures,
            "non_retryable": self.non_retryable,
            "retry_attempts": self.retry_attempts,
            "retry_success_rate": round(self.retry_success_rate, 4),
            "by_category": dict(self.by_category),
        }


@dataclass
class ReviewMetrics:
    """Counts documents and fields routed to human review."""

    documents_reviewed: int = 0
    fields_flagged: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    def record_document(self, flagged_fields: int, reasons: list[str]) -> None:
        if flagged_fields:
            self.documents_reviewed += 1
            self.fields_flagged += flagged_fields
            for reason in reasons:
                self.by_reason[reason] = self.by_reason.get(reason, 0) + 1

    def to_dict(self) -> dict:
        return {
            "documents_reviewed": self.documents_reviewed,
            "fields_flagged": self.fields_flagged,
            "by_reason": dict(self.by_reason),
        }


@dataclass
class SLACalculator:
    """Compute sequential vs batch processing time and SLA compliance.

    Parameters
    ----------
    avg_extraction_seconds:
        Mean wall-clock time for a single extraction call.
    document_count:
        Number of documents in the workload.
    batch_concurrency:
        Effective parallelism of the Message Batches API for this workload.
    retry_rate:
        Fraction of documents that trigger one extra (retry) call.
    sla_seconds:
        The contractual deadline to compare against.
    """

    avg_extraction_seconds: float = 2.0
    document_count: int = 100
    batch_concurrency: int = 50
    retry_rate: float = 0.10
    sla_seconds: float = 3600.0

    @property
    def sequential_seconds(self) -> float:
        return self.avg_extraction_seconds * self.document_count

    @property
    def retry_overhead_seconds(self) -> float:
        """Extra time from retrying a fraction of documents once."""
        return self.avg_extraction_seconds * self.document_count * self.retry_rate

    @property
    def batch_seconds(self) -> float:
        """Idealized batch time: documents fan out across ``batch_concurrency``.

        Retries add one more wave proportional to ``retry_rate``.
        """
        waves = self.document_count / max(self.batch_concurrency, 1)
        retry_waves = (self.document_count * self.retry_rate) / max(
            self.batch_concurrency, 1
        )
        return self.avg_extraction_seconds * (waves + retry_waves)

    def meets_sla(self, mode: str = "batch") -> bool:
        total = self.batch_seconds if mode == "batch" else self.sequential_seconds
        return total <= self.sla_seconds

    def to_dict(self) -> dict:
        return {
            "assumptions": {
                "avg_extraction_seconds": self.avg_extraction_seconds,
                "document_count": self.document_count,
                "batch_concurrency": self.batch_concurrency,
                "retry_rate": self.retry_rate,
                "sla_seconds": self.sla_seconds,
            },
            "sequential_seconds": round(self.sequential_seconds, 2),
            "batch_seconds": round(self.batch_seconds, 2),
            "retry_overhead_seconds": round(self.retry_overhead_seconds, 2),
            "sequential_meets_sla": self.meets_sla("sequential"),
            "batch_meets_sla": self.meets_sla("batch"),
        }


__all__ = ["RetryMetrics", "ReviewMetrics", "SLACalculator"]

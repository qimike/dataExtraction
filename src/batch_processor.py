"""Batch processing strategy (Step 4).

Two responsibilities:

1. **Message Batches API workflow** — submit many extractions as one batch,
   poll for completion, collect results keyed by ``custom_id``, and resubmit
   the failures. ``submit_batch`` / ``poll_batch`` / ``collect_results`` wrap
   the live Anthropic Batches endpoints; ``simulate_batch`` runs the same
   control flow deterministically (no API key) so the workflow is testable and
   the repo is reproducible.

2. **Chunking strategy for oversized documents** — split a document that
   exceeds a token budget into overlapping chunks, extract each, and recombine.

Cost note: the Batches API runs at 50% of standard price and most batches
finish within an hour — ideal for non-latency-sensitive bulk extraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.extractor import (
    MAX_TOKENS,
    MODEL,
    TOOL_NAME,
    build_messages,
    build_system_prompt,
    build_tool,
    ExtractionRequest,
)

# --- Chunking configuration (documented in the README) --------------------
CHUNK_SIZE_CHARS = 8000      # ~2k tokens per chunk; well under the model limit
CHUNK_OVERLAP_CHARS = 800    # 10% overlap so entities spanning a boundary survive


@dataclass
class BatchItem:
    """One request in a batch, tracked by its ``custom_id``."""

    custom_id: str
    document_text: str
    include_few_shot: bool = True


def build_batch_requests(items: list[BatchItem]) -> list[dict]:
    """Build the ``requests`` payload for ``client.messages.batches.create``.

    Each request mirrors the single-document extractor: same model, tool, and
    forced ``tool_choice``, tagged with the item's ``custom_id`` so results can
    be matched back to their source document.
    """
    requests = []
    for item in items:
        req = ExtractionRequest(
            document_id=item.custom_id,
            document_text=item.document_text,
            include_few_shot=item.include_few_shot,
        )
        requests.append(
            {
                "custom_id": item.custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "thinking": {"type": "adaptive"},
                    "system": build_system_prompt(item.include_few_shot),
                    "tools": [build_tool()],
                    "tool_choice": {"type": "tool", "name": TOOL_NAME},
                    "messages": build_messages(req),
                },
            }
        )
    return requests


def submit_batch(items: list[BatchItem], client: Any) -> str:
    """Submit a batch and return its id."""
    batch = client.messages.batches.create(requests=build_batch_requests(items))
    return batch.id


def poll_batch(batch_id: str, client: Any, *, interval: float = 60.0) -> Any:
    """Poll until the batch ends, then return the final batch object."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        time.sleep(interval)


def collect_results(batch_id: str, client: Any) -> dict[str, dict]:
    """Collect results keyed by ``custom_id``.

    Returns a mapping ``custom_id -> {"status": ..., "extraction"|"error": ...}``.
    Succeeded entries carry the raw tool input; everything else carries a
    status so the caller can decide what to resubmit.
    """
    results: dict[str, dict] = {}
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        kind = result.result.type
        if kind == "succeeded":
            extraction = _extract_tool_input(result.result.message)
            results[cid] = {"status": "succeeded", "extraction": extraction}
        elif kind == "errored":
            results[cid] = {
                "status": "errored",
                "error": getattr(result.result.error, "type", "unknown"),
            }
        else:  # canceled | expired
            results[cid] = {"status": kind}
    return results


def find_failures(results: dict[str, dict]) -> list[str]:
    """Return the ``custom_id``s that did not succeed and should be resubmitted."""
    return [cid for cid, r in results.items() if r.get("status") != "succeeded"]


def _extract_tool_input(message: Any) -> Optional[dict]:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
            return dict(block.input)
    return None


# --- Deterministic simulation (no API key required) ------------------------


@dataclass
class SimulatedBatchReport:
    """Outcome of a simulated batch run."""

    submitted: int
    succeeded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    resubmitted: list[str] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "submitted": self.submitted,
            "succeeded_count": len(self.succeeded),
            "failed_count": len(self.failed),
            "failed_custom_ids": self.failed,
            "resubmitted": self.resubmitted,
            "recovered": self.recovered,
            "final_success_count": len(self.succeeded) + len(self.recovered),
        }


def simulate_batch(
    items: list[BatchItem],
    *,
    forced_failures: Optional[set[str]] = None,
    recover_on_resubmit: bool = True,
) -> SimulatedBatchReport:
    """Run the submit -> collect -> resubmit-failures control flow offline.

    ``forced_failures`` lets a caller (or test) mark specific ``custom_id``s as
    failing on the first pass — mirroring the doc_017 / doc_048 / doc_081
    example in the spec. On resubmission those are recovered (the default),
    demonstrating failure recovery by ``custom_id``.
    """
    forced_failures = forced_failures or set()
    report = SimulatedBatchReport(submitted=len(items))

    for item in items:
        if item.custom_id in forced_failures:
            report.failed.append(item.custom_id)
        else:
            report.succeeded.append(item.custom_id)

    if report.failed:
        report.resubmitted = list(report.failed)
        if recover_on_resubmit:
            report.recovered = list(report.failed)

    return report


# --- Chunking strategy for oversized documents -----------------------------


def chunk_document(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split a long document into overlapping chunks.

    The overlap (``overlap`` characters carried from the end of one chunk to the
    start of the next) ensures an entity that straddles a boundary — an author
    line split mid-sentence, a DOI broken across a page break — still appears
    intact in at least one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def recombine_chunk_extractions(extractions: list[dict]) -> dict:
    """Recombine per-chunk extractions into one (highest-confidence-wins).

    Strategy: for each field, keep the candidate with the highest confidence
    across chunks. This favors the chunk that actually contained the field over
    chunks that guessed. ``overall_confidence`` is averaged across chunks.
    """
    if not extractions:
        return {}
    if len(extractions) == 1:
        return extractions[0]

    combined: dict[str, Any] = {}
    confidences: list[float] = []

    for extraction in extractions:
        if "overall_confidence" in extraction:
            confidences.append(extraction["overall_confidence"])
        for field_name, value in extraction.items():
            if field_name == "overall_confidence":
                continue
            if not isinstance(value, dict) or "confidence" not in value:
                continue
            current = combined.get(field_name)
            if current is None or value["confidence"] > current["confidence"]:
                # Prefer non-null values when confidences tie.
                if (
                    current is None
                    or value["value"] is not None
                    or current["value"] is None
                ):
                    combined[field_name] = value

    combined["overall_confidence"] = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )
    return combined


__all__ = [
    "CHUNK_SIZE_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "BatchItem",
    "build_batch_requests",
    "submit_batch",
    "poll_batch",
    "collect_results",
    "find_failures",
    "SimulatedBatchReport",
    "simulate_batch",
    "chunk_document",
    "recombine_chunk_extractions",
]

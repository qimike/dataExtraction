"""End-to-end pipeline runner.

Wires the components together for the full flow:

    document -> extract -> validate -> retry -> confidence routing -> review

and writes the three deliverable artifacts to ``output/``:

* ``extracted_results.json``   — every successfully validated extraction
* ``validation_failures.json`` — documents that failed (with captured context)
* ``review_queue.json``        — low-confidence fields routed to a human

Run modes
---------
* **Offline (default).** Uses a deterministic, hand-authored extractor that
  emulates what Claude returns for the six sample documents — including a
  retry-recoverable enum mismatch and a non-retryable missing-title case. This
  makes the repo fully reproducible (``python -m src.pipeline``) with no API
  key, and gives the README its concrete numbers.
* **Live (``--live``).** Calls Claude (``claude-opus-4-8``) via the real
  extractor. Requires ``ANTHROPIC_API_KEY``.

Usage::

    python -m src.pipeline            # offline, writes output/*.json
    python -m src.pipeline --live     # hit the Anthropic API
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

# Path shim so `python src/pipeline.py` and `-m src.pipeline` both work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence_router import DEFAULT_THRESHOLD, build_review_queue  # noqa: E402
from src.extractor import ExtractionRequest, extract  # noqa: E402
from src.metrics import ReviewMetrics, RetryMetrics, SLACalculator  # noqa: E402
from src.retry_handler import process_document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_documents"
OUTPUT_DIR = ROOT / "output"

# Map sample files to stable document ids, in processing order.
DOCUMENTS = [
    ("doc_001", "document_01.txt"),
    ("doc_002", "document_02.txt"),
    ("doc_003", "document_03.txt"),
    ("doc_004", "document_missing_fields.txt"),
    ("doc_005", "document_table_format.txt"),
    ("doc_006", "document_bibliography_format.txt"),
]


def _cf(value: Any, confidence: float) -> dict:
    """Shorthand for a confidence-wrapped field."""
    return {"value": value, "confidence": confidence}


# --- Offline simulated extractions -----------------------------------------
# Each entry has a "first" payload and, when a retry is expected, a "retry"
# payload returned on the second attempt. This is what a real model would emit;
# it lets the loop exercise success, retry-recovery, and non-retryable failure.
SIMULATED: dict[str, dict[str, dict]] = {
    "doc_001": {
        "first": {
            "document_id": _cf("doc_001", 1.0),
            "title": _cf("Retrieval-Augmented Generation for Clinical Question Answering", 0.97),
            "document_type": _cf("research_paper", 0.96),
            "summary": _cf("A RAG system that grounds clinical question answering in retrieved EHR passages, improving accuracy and reducing unsupported claims.", 0.91),
            "other_document_type_detail": None,
            "publication_date": _cf("2024-01-01", 0.55),
            "author": _cf("Patel, A., Nguyen, T., Rossi, M.", 0.93),
            "organization": _cf("Conference on Health Informatics", 0.7),
            "email": _cf("a.patel@meditech-research.org", 0.98),
            "phone_number": _cf(None, 0.99),
            "doi": _cf("10.1109/HCI.2024.0042", 0.98),
            "citation_count": _cf(None, 0.95),
            "overall_confidence": 0.89,
        }
    },
    "doc_002": {
        # First attempt picks an enum value outside the allowed set.
        "first": {
            "document_id": _cf("doc_002", 1.0),
            "title": _cf("Annual Sustainability Report 2023", 0.96),
            "document_type": _cf("sustainability_report", 0.7),
            "summary": _cf("GreenGrid Energy Cooperative's 2023 sustainability report covering emissions reductions, member generation growth, and 2030 net-zero plans.", 0.9),
            "other_document_type_detail": None,
            "publication_date": _cf("2024-03-01", 0.8),
            "author": _cf("Office of Corporate Responsibility", 0.82),
            "organization": _cf("GreenGrid Energy Cooperative", 0.95),
            "email": _cf("sustainability@greengrid.coop", 0.97),
            "phone_number": _cf("+1 (415) 555-0148", 0.96),
            "doi": _cf(None, 0.99),
            "citation_count": _cf(None, 0.99),
            "overall_confidence": 0.9,
        },
        # Retry corrects the enum to an allowed value.
        "retry": {
            "document_id": _cf("doc_002", 1.0),
            "title": _cf("Annual Sustainability Report 2023", 0.96),
            "document_type": _cf("report", 0.92),
            "summary": _cf("GreenGrid Energy Cooperative's 2023 sustainability report covering emissions reductions, member generation growth, and 2030 net-zero plans.", 0.9),
            "other_document_type_detail": None,
            "publication_date": _cf("2024-03-01", 0.8),
            "author": _cf("Office of Corporate Responsibility", 0.82),
            "organization": _cf("GreenGrid Energy Cooperative", 0.95),
            "email": _cf("sustainability@greengrid.coop", 0.97),
            "phone_number": _cf("+1 (415) 555-0148", 0.96),
            "doi": _cf(None, 0.99),
            "citation_count": _cf(None, 0.99),
            "overall_confidence": 0.91,
        },
    },
    "doc_003": {
        "first": {
            "document_id": _cf("doc_003", 1.0),
            "title": _cf("Invoice INV-2024-3391", 0.88),
            "document_type": _cf("invoice", 0.97),
            "summary": _cf("A consulting invoice from Brightpath Consulting LLC to Harborview Logistics totaling $10,397.56, due 2024-06-11.", 0.92),
            "other_document_type_detail": None,
            "publication_date": _cf("2024-05-12", 0.85),
            "author": _cf(None, 0.9),
            "organization": _cf("Brightpath Consulting LLC", 0.94),
            "email": _cf("billing@brightpath-consulting.com", 0.98),
            "phone_number": _cf("(212) 555-0193", 0.96),
            "doi": _cf(None, 0.99),
            "citation_count": _cf(None, 0.99),
            "overall_confidence": 0.93,
        }
    },
    "doc_004": {
        # The notes file has no real title -> model returns null for a required
        # field -> MISSING_INFORMATION (non-retryable). Routes to failures.
        "first": {
            "document_id": _cf("doc_004", 1.0),
            "title": _cf(None, 0.2),
            "document_type": _cf("other", 0.4),
            "summary": _cf("Informal offsite notes about the onboarding flow; no decisions recorded.", 0.6),
            "other_document_type_detail": _cf("informal notes", 0.4),
            "publication_date": _cf(None, 0.95),
            "author": _cf(None, 0.95),
            "organization": _cf(None, 0.95),
            "email": _cf(None, 0.99),
            "phone_number": _cf(None, 0.99),
            "doi": _cf(None, 0.99),
            "citation_count": _cf(None, 0.99),
            "overall_confidence": 0.45,
        }
    },
    "doc_005": {
        "first": {
            "document_id": _cf("doc_005", 1.0),
            "title": _cf("Master Services Agreement", 0.98),
            "document_type": _cf("contract", 0.95),
            "summary": _cf("A master services agreement governing contract manufacturing services for Vertex Manufacturing Group, including confidentiality and termination clauses.", 0.9),
            "other_document_type_detail": None,
            "publication_date": _cf("2024-01-15", 0.9),
            "author": _cf("Dana Whitfield, General Counsel", 0.95),
            "organization": _cf("Vertex Manufacturing Group", 0.96),
            "email": _cf("contracts@vertex-mfg.com", 0.97),
            "phone_number": _cf("+1 (650) 555-0177", 0.96),
            "doi": _cf(None, 0.99),
            "citation_count": _cf(None, 0.99),
            "overall_confidence": 0.95,
        }
    },
    "doc_006": {
        "first": {
            "document_id": _cf("doc_006", 1.0),
            "title": _cf("Sparse Mixture-of-Experts Routing for Long-Context Models", 0.97),
            "document_type": _cf("research_paper", 0.95),
            "summary": _cf("A research paper studying load-balanced top-k routing for sparse mixture-of-experts layers in long-context transformers.", 0.9),
            "other_document_type_detail": None,
            "publication_date": _cf(None, 0.7),
            "author": _cf("J. Okafor, R. Mehta, L. Persson", 0.94),
            "organization": _cf("Institute for Scalable Computing", 0.62),
            "email": _cf(None, 0.97),
            "phone_number": _cf(None, 0.99),
            "doi": _cf("10.48550/arXiv.2406.01234", 0.96),
            "citation_count": _cf(87, 0.93),
            "overall_confidence": 0.88,
        }
    },
}


def make_offline_extractor() -> Callable[[ExtractionRequest], dict]:
    """Return an ``extract_fn`` that serves the simulated payloads.

    Returns the ``retry`` payload when the request carries retry context and one
    is defined; otherwise the ``first`` payload.
    """

    def _extract(req: ExtractionRequest) -> dict:
        entry = SIMULATED[req.document_id]
        if req.previous_extraction is not None and "retry" in entry:
            return json.loads(json.dumps(entry["retry"]))
        return json.loads(json.dumps(entry["first"]))

    return _extract


def run(live: bool = False, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Run the pipeline over the sample documents and write output files."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if live:
        extract_fn = lambda req: extract(req)  # noqa: E731
    else:
        extract_fn = make_offline_extractor()

    retry_metrics = RetryMetrics()
    review_metrics = ReviewMetrics()

    extracted_results: list[dict] = []
    validation_failures: list[dict] = []
    successful_results = []

    for doc_id, filename in DOCUMENTS:
        text = (SAMPLE_DIR / filename).read_text(encoding="utf-8")
        record = process_document(
            doc_id, text, extract_fn=extract_fn, metrics=retry_metrics
        )
        if record.succeeded and record.result is not None:
            successful_results.append(record.result)
            extracted_results.append(
                {
                    "document_id": doc_id,
                    "source_file": filename,
                    "attempts": record.attempts,
                    "retried": record.retried,
                    "extraction": record.result.model_dump(mode="json"),
                }
            )
        else:
            validation_failures.append(
                {
                    "document_id": doc_id,
                    "source_file": filename,
                    **record.to_dict(),
                }
            )

    # Confidence routing across the successful extractions.
    review_queue = build_review_queue(successful_results, threshold)
    # Roll up review metrics per document.
    from src.confidence_router import route_document

    for result in successful_results:
        routing = route_document(result, threshold)
        review_metrics.record_document(len(routing.items), routing.reasons)

    # SLA analysis for a 100-document workload.
    sla = SLACalculator(
        avg_extraction_seconds=2.0,
        document_count=100,
        batch_concurrency=50,
        retry_rate=retry_metrics.retry_attempts / max(len(DOCUMENTS), 1),
        sla_seconds=3600.0,
    )

    # Write artifacts.
    _write(OUTPUT_DIR / "extracted_results.json", extracted_results)
    _write(OUTPUT_DIR / "validation_failures.json", validation_failures)
    _write(OUTPUT_DIR / "review_queue.json", review_queue)

    summary = {
        "documents_processed": len(DOCUMENTS),
        "succeeded": len(extracted_results),
        "failed": len(validation_failures),
        "retry_metrics": retry_metrics.to_dict(),
        "review_metrics": review_metrics.to_dict(),
        "sla": sla.to_dict(),
        "confidence_threshold": threshold,
    }
    return summary


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _print_summary(summary: dict) -> None:
    print("\n=== Pipeline summary ===")
    print(f"Documents processed : {summary['documents_processed']}")
    print(f"Succeeded           : {summary['succeeded']}")
    print(f"Failed              : {summary['failed']}")
    rm = summary["retry_metrics"]
    print(
        f"Retries             : {rm['retry_successes']} ok / "
        f"{rm['retry_failures']} failed "
        f"(rate {rm['retry_success_rate']:.0%}), "
        f"non-retryable {rm['non_retryable']}"
    )
    vm = summary["review_metrics"]
    print(
        f"Human review        : {vm['documents_reviewed']} docs, "
        f"{vm['fields_flagged']} fields"
    )
    sla = summary["sla"]
    print(
        f"SLA (100 docs)      : sequential {sla['sequential_seconds']}s, "
        f"batch {sla['batch_seconds']}s, "
        f"batch meets SLA: {sla['batch_meets_sla']}"
    )
    print("Artifacts written to output/.\n")


if __name__ == "__main__":
    live = "--live" in sys.argv
    result = run(live=live)
    _print_summary(result)

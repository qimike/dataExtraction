"""Claude-backed structured extractor (Step 1: Tool Use for Structured Output).

Sends a document to Claude and forces it to call a single ``extract_document``
tool whose ``input_schema`` is generated directly from the Pydantic
:class:`ExtractionResult`. Forcing the tool guarantees the response arrives as
a structured ``tool_use`` block rather than free-text we would have to parse.

The extractor is deliberately thin: it returns the *raw* tool input ``dict``.
Validation lives in ``src/validator.py`` and the retry orchestration in
``src/retry_handler.py`` so each concern is independently testable.

Model & API choices (per the Anthropic Claude API guidance):
* Model ``claude-opus-4-8`` — the current, most capable Opus-tier model.
* Adaptive thinking (``thinking={"type": "adaptive"}``) for reliable reasoning
  on messy documents.
* ``tool_choice`` forces the extraction tool so output is always structured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from schemas.extraction_schema import extraction_tool_schema

MODEL = "claude-opus-4-8"
TOOL_NAME = "extract_document"
MAX_TOKENS = 4096

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    path = _PROMPT_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def build_system_prompt(include_few_shot: bool = True) -> str:
    """Assemble the system prompt from the prompt files.

    Few-shot examples are appended after the core instructions; toggling them
    off lets us A/B the with/without-few-shot comparison required by Step 3.
    """
    parts = [_load_prompt("extraction_prompt.md")]
    if include_few_shot:
        few_shot = _load_prompt("few_shot_examples.md")
        if few_shot:
            parts.append("\n\n# Few-Shot Examples\n\n" + few_shot)
    return "\n".join(p for p in parts if p)


def build_tool() -> dict:
    """The Anthropic tool definition driving structured output."""
    return {
        "name": TOOL_NAME,
        "description": (
            "Extract structured metadata from the supplied document. Call this "
            "tool exactly once with your best extraction. Return null for any "
            "field whose information is not present in the document — never "
            "fabricate values."
        ),
        "input_schema": extraction_tool_schema(),
    }


@dataclass
class ExtractionRequest:
    """Inputs for one extraction (initial or retry)."""

    document_id: str
    document_text: str
    include_few_shot: bool = True
    # Retry context (populated only on a retry attempt):
    previous_extraction: Optional[dict[str, Any]] = None
    validation_error: Optional[str] = None


def build_messages(req: ExtractionRequest) -> list[dict]:
    """Construct the user-turn messages for an extraction request.

    On a retry, the prior (invalid) extraction and the validation error are
    embedded so the model can correct its own output — the core of the
    validation-retry loop in Step 2.
    """
    user_text = (
        f"Document id: {req.document_id}\n\n"
        f"--- BEGIN DOCUMENT ---\n{req.document_text}\n--- END DOCUMENT ---\n\n"
        "Extract the structured metadata by calling the extract_document tool."
    )

    if req.previous_extraction is not None and req.validation_error:
        import json

        user_text += (
            "\n\nYour previous extraction failed validation. "
            "Fix only what the error describes and resubmit via the tool.\n\n"
            f"Previous extraction:\n{json.dumps(req.previous_extraction, indent=2)}\n\n"
            f"Validation error:\n{req.validation_error}"
        )

    return [{"role": "user", "content": user_text}]


def extract(req: ExtractionRequest, client: Optional[Any] = None) -> dict[str, Any]:
    """Run one extraction against Claude and return the raw tool input.

    Parameters
    ----------
    req:
        The document (and optional retry context) to extract.
    client:
        An ``anthropic.Anthropic`` instance. If omitted, one is constructed
        from the environment (``ANTHROPIC_API_KEY``). Injecting a client makes
        the function trivially mockable in tests.

    Returns
    -------
    dict
        The ``input`` of the model's ``tool_use`` block — i.e. the raw
        extraction, not yet validated.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=build_system_prompt(include_few_shot=req.include_few_shot),
        tools=[build_tool()],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=build_messages(req),
    )

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
            return dict(block.input)

    raise RuntimeError("Model did not return an extract_document tool_use block.")


__all__ = [
    "MODEL",
    "TOOL_NAME",
    "ExtractionRequest",
    "build_system_prompt",
    "build_tool",
    "build_messages",
    "extract",
]

"""Versioned prompts. A prompt is immutable once released: change the text means
bump the version. Each extraction run records ``prompt_id``, ``version`` and the
SHA-256 of the full template plus response schema, so a run can be reproduced
exactly and silent edits are detectable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from veritas.extraction.schema import response_schema
from veritas.extraction.sources import SourceUnit
from veritas.llm.base import ChatMessage


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    version: str
    system: str
    user_template: str  # must contain {sources}

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            {"system": self.system, "user": self.user_template, "schema": response_schema()},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def messages(self, units: list[SourceUnit]) -> tuple[ChatMessage, ...]:
        sources = "\n\n".join(u.render() for u in units)
        return (
            ChatMessage("system", self.system),
            ChatMessage("user", self.user_template.format(sources=sources)),
        )


_SYSTEM_V1 = """\
You extract atomic factual claims and named entities from document sources.
You return JSON that matches the given schema. You never invent information.

CLAIMS
- One claim per atomic fact. A source may contain several claims; extract each.
- Only facts explicitly stated in the source. No inference, no outside knowledge.
- subject: the entity the fact is about, written exactly as in the source or its context.
- predicate: a short snake_case attribute WITHOUT modality words, e.g. budget,
  expenditure, cost, completion_date, start_date, status, supervisor, host_institution.
- modality: how the source qualifies the value:
    proposed = proposed, requested or planned
    approved = approved, allocated or sanctioned
    actual   = actually spent, recorded or achieved
    forecast = forecast, projected or expected
    unknown  = none of the above is stated
  Never guess modality. "proposed budget" and "approved budget" are different claims.
- object_type and value:
    money   -> value = plain digits, no separators ("LKR 5 million" -> "5000000"),
               currency = ISO 4217 code that appears in the source (LKR, USD, ...)
    number  -> plain digits; unit if stated, else null
    percent -> plain digits; unit = "percent"
    date    -> "YYYY", "YYYY-MM" or "YYYY-MM-DD"
    text    -> the value as written (e.g. "Completed", a person's name)
    boolean -> "true" or "false"
- valid_from / valid_to: the period the fact applies to, only if stated in the source
  or its context; else null. time_expression: the exact words stating it, else null.
- negated: true only if the source says the fact did NOT happen.
- evidence_quotes: 1-4 exact, contiguous substrings copied from the source that state
  the fact. For table rows, quote the cell values exactly as shown.
- confidence: 0-1, how certain you are that you read the source correctly.

ENTITIES
- Every organization, project, person, program or location named in the sources.
- name: exactly as written. aliases: other names for the SAME entity that appear in
  the sources (e.g. an acronym defined in parentheses); else [].

If a source contains no claims, return no claims for it. Use only the given source ids."""

_USER_V1 = """\
Sources:

{sources}

Return the JSON object with "claims" and "entities"."""

CLAIM_EXTRACTION_V1 = PromptTemplate(
    prompt_id="claim_entity_extraction",
    version="1.0.0",
    system=_SYSTEM_V1,
    user_template=_USER_V1,
)

# 1.1.0 (2026-11-28): "or its context" removed for subject and period; currency and
# period must be stated in the cited evidence itself, next to the value. 1.0.0 is kept
# unchanged so earlier runs stay reproducible from their recorded prompt version.
_SYSTEM_V1_1 = (
    _SYSTEM_V1.replace(
        "- subject: the entity the fact is about, written exactly as in the source or its context.",
        "- subject: the entity the fact is about, written exactly as in the quoted sentence or\n"
        "  table row (not only in a heading).",
    )
    .replace(
        """    money   -> value = plain digits, no separators ("LKR 5 million" -> "5000000"),
               currency = ISO 4217 code that appears in the source (LKR, USD, ...)""",
        """    money   -> value = plain digits, no separators ("LKR 5 million" -> "5000000"),
               currency = ISO 4217 code written next to that amount in your
               evidence_quotes, or in the column header of the cited value cell;
               null if the quotes do not state it or only use an ambiguous symbol
               such as '$' or 'Rs.'""",
    )
    .replace(
        """- valid_from / valid_to: the period the fact applies to, only if stated in the source
  or its context; else null. time_expression: the exact words stating it, else null.""",
        """- valid_from / valid_to: the period the fact applies to, only if it is stated in the
  same quoted sentence or table row as the fact (never from headings, titles or sheet
  names); else null. Keep the stated precision: "2025" stays "2025".
  time_expression: the exact words stating the period, included in evidence_quotes;
  else null.""",
    )
)

CLAIM_EXTRACTION_V1_1 = PromptTemplate(
    prompt_id="claim_entity_extraction",
    version="1.1.0",
    system=_SYSTEM_V1_1,
    user_template=_USER_V1,
)

PROMPTS: dict[tuple[str, str], PromptTemplate] = {
    (p.prompt_id, p.version): p for p in (CLAIM_EXTRACTION_V1, CLAIM_EXTRACTION_V1_1)
}
DEFAULT_PROMPT = CLAIM_EXTRACTION_V1_1


def get_prompt(version: str, prompt_id: str = "claim_entity_extraction") -> PromptTemplate:
    try:
        return PROMPTS[(prompt_id, version)]
    except KeyError:
        known = sorted(v for pid, v in PROMPTS if pid == prompt_id)
        raise ValueError(f"unknown prompt {prompt_id}@{version}; known versions: {known}") from None


def retry_message(error: str) -> ChatMessage:
    return ChatMessage(
        "user",
        "Your previous response was rejected because it did not match the required "
        f"format:\n{error}\nReturn the complete corrected JSON object only.",
    )

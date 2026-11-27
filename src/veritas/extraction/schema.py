"""The JSON contract the model must return (validated strictly, then grounded).

This is deliberately *not* the ``Claim`` contract: the model proposes candidates,
and ``extraction.validator`` decides what becomes a ``Claim``.

Core fields are required; a response missing one is invalid and retried. Optional
context (unit, currency, period, negation, aliases) defaults to null/false/[] when
omitted: JSON-mode models (e.g. qwen3:4b-instruct-2507 with format="json") drop
null fields, and an absent optional value carries no claim. Unknown fields are
always rejected.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from veritas.core.contracts import EntityType, Modality, ObjectType


class _ModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LLMClaim(_ModelOutput):
    source_id: str = Field(description="Id of the source the claim comes from, e.g. 'S1'")
    subject: str = Field(min_length=1, description="Entity the claim is about, as written")
    predicate: str = Field(min_length=1, description="Attribute in snake_case, without modality")
    object_type: ObjectType
    value: str = Field(
        min_length=1, description="Digits only for numbers; YYYY[-MM[-DD]] for dates"
    )
    unit: str | None = Field(
        default=None, description="Non-currency unit, e.g. 'percent'; else null"
    )
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. 'LKR'; else null")
    modality: Modality
    valid_from: str | None = Field(default=None, description="YYYY, YYYY-MM or YYYY-MM-DD, or null")
    valid_to: str | None = Field(default=None, description="YYYY, YYYY-MM or YYYY-MM-DD, or null")
    time_expression: str | None = Field(
        default=None, description="Exact source words stating the time"
    )
    negated: bool = False
    evidence_quotes: list[str] = Field(min_length=1, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)


class LLMEntity(_ModelOutput):
    source_id: str
    name: str = Field(min_length=1, description="Exact text as written in the source")
    entity_type: EntityType
    aliases: list[str] = Field(
        default_factory=list, description="Other names for it that appear in the sources"
    )


class LLMExtraction(_ModelOutput):
    claims: list[LLMClaim]
    entities: list[LLMEntity]


def response_schema() -> dict[str, Any]:
    """JSON Schema passed to the provider for constrained decoding."""
    return LLMExtraction.model_json_schema()

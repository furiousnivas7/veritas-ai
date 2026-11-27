"""Core data contracts shared by every VERITAS module.

Design rules enforced here (see docs/data_model.md):

1. A ``Claim`` cannot exist without at least one ``Evidence``.
2. ``Evidence`` must carry a *complete* ``SourceLocation`` for its format:
   PDF -> page number; XLSX -> sheet + cell/range; DOCX -> element id + structure.
3. A ``DocumentSpan`` may lack a location, but then it must say why
   (``missing_provenance``). Such spans can never be cited as evidence.
4. Extraction confidence (how sure we are the claim was read correctly) is kept
   separate from verification confidence (how sure we are the claim is true).
5. Abstention is explicit: ``INSUFFICIENT_EVIDENCE`` and ``REQUIRES_HUMAN_REVIEW``
   are first-class verdicts, and abstaining requires a reason.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256 = r"^[0-9a-f]{64}$"
_PARTIAL_DATE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")
_CELL_RANGE = re.compile(r"^[A-Z]{1,3}[1-9]\d*(:[A-Z]{1,3}[1-9]\d*)?$")
_SNAKE = r"^[a-z][a-z0-9_]*$"


def utcnow() -> datetime:
    return datetime.now(UTC)


def parse_partial_date(value: str) -> tuple[int, int | None, int | None] | None:
    """'2025' / '2025-02' / '2024-02-29' -> (year, month, day); None if malformed or
    not a real calendar date ('2025-02-30', '2025-13', and a day without a month)."""
    m = _PARTIAL_DATE.match(value)
    if not m:
        return None
    year = int(m.group(1))
    month = int(m.group(2)) if m.group(2) else None
    day = int(m.group(3)) if m.group(3) else None
    try:
        date(year, 1 if month is None else month, 1 if day is None else day)
    except ValueError:  # month 0/13, day 0/32, 30 February, 29 February in non-leap years
        return None
    return year, month, day


class _Strict(BaseModel):
    # LLM output is validated against these models, so unknown fields are an error.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- enums


class DocumentFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"


class DocumentStatus(StrEnum):
    REGISTERED = "registered"
    PARSED = "parsed"
    FAILED = "failed"


class SpanKind(StrEnum):
    TITLE = "title"
    HEADING = "heading"
    TEXT = "text"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    OTHER = "other"


class Modality(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    ACTUAL = "actual"
    FORECAST = "forecast"
    UNKNOWN = "unknown"


class ObjectType(StrEnum):
    """What kind of value a claim's object is."""

    MONEY = "money"  # Quantity with currency
    NUMBER = "number"  # Quantity with a non-currency unit (or none)
    PERCENT = "percent"  # Quantity with unit 'percent'
    DATE = "date"  # str, YYYY / YYYY-MM / YYYY-MM-DD
    TEXT = "text"  # str
    BOOLEAN = "boolean"  # bool


class EntityType(StrEnum):
    ORGANIZATION = "organization"
    PROJECT = "project"
    PERSON = "person"
    LOCATION = "location"
    PROGRAM = "program"
    OTHER = "other"


class MatchStatus(StrEnum):
    """Outcome of comparing an entity mention with an existing entity."""

    MERGED = "merged"  # strong evidence: exact normalised name or explicit alias definition
    NEEDS_REVIEW = "needs_review"  # plausible but not proven; never merged automatically
    REJECTED = "rejected"  # reviewed and found to be different entities


class ConflictType(StrEnum):
    """How two claims disagree. Only assigned when the verdict is CONTRADICTED."""

    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    ENTITY = "entity"
    LOGICAL = "logical"


class Verdict(StrEnum):
    SUPPORTED = "supported"
    """Claim vs evidence: the evidence entails the claim."""
    REFUTED = "refuted"
    """Claim vs evidence: the evidence states the opposite of the claim."""
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    """Abstention: nothing found supports or refutes the claim."""
    CONTRADICTED = "contradicted"
    """Claim vs claim: same subject, predicate, modality and period; both cannot be true."""
    CONTEXTUALLY_DIFFERENT = "contextually_different"
    """Claim vs claim: values differ but contexts differ (e.g. proposed vs approved)."""
    REQUIRES_HUMAN_REVIEW = "requires_human_review"
    """Abstention: the system cannot decide with enough confidence."""


ABSTENTION_VERDICTS = frozenset({Verdict.INSUFFICIENT_EVIDENCE, Verdict.REQUIRES_HUMAN_REVIEW})
CLAIM_VERDICTS = frozenset(
    {
        Verdict.SUPPORTED,
        Verdict.REFUTED,
        Verdict.INSUFFICIENT_EVIDENCE,
        Verdict.REQUIRES_HUMAN_REVIEW,
    }
)
CONFLICT_VERDICTS = frozenset(
    {Verdict.CONTRADICTED, Verdict.CONTEXTUALLY_DIFFERENT, Verdict.REQUIRES_HUMAN_REVIEW}
)


class RelationType(StrEnum):
    MENTIONS = "mentions"
    ASSERTS = "asserts"
    SUBJECT_OF = "subject_of"
    OBJECT_OF = "object_of"
    SUPPORTED_BY = "supported_by"
    REFUTED_BY = "refuted_by"
    CONTRADICTS = "contradicts"
    SAME_AS = "same_as"
    DERIVED_FROM = "derived_from"


# ------------------------------------------------------------------- documents


class Document(_Strict):
    """One registered file. Re-uploading changed content under the same filename
    creates a new Document with ``version`` + 1 and ``supersedes`` pointing back."""

    document_id: str = Field(pattern=r"^DOC-[0-9a-f]{12}$")
    filename: str = Field(min_length=1)
    format: DocumentFormat
    mime_type: str
    sha256: str = Field(pattern=_SHA256)
    size_bytes: int = Field(gt=0)
    version: int = Field(ge=1)
    supersedes: str | None = None
    status: DocumentStatus = DocumentStatus.REGISTERED
    error: str | None = None
    registered_at: datetime = Field(default_factory=utcnow)


class SourceLocation(_Strict):
    """A complete, citable location. Validation depends on the document format."""

    format: DocumentFormat
    element_id: str = Field(min_length=1, description="Docling self_ref, e.g. '#/texts/12'")

    # PDF: 1-based physical page. page_label is the printed label, only when the
    # parser exposes it (Docling 2.154 does not); never derived by guessing.
    page_number: int | None = Field(default=None, ge=1)
    page_label: str | None = None

    # XLSX: sheet + A1-style cell or range. Spreadsheets have no pages.
    sheet_name: str | None = None
    cell_range: str | None = None

    # DOCX (and PDF) structure: headings above the element, and table coordinates
    # (0-based, relative to the table) for table cells.
    heading_path: list[str] = Field(default_factory=list)
    table_ref: str | None = Field(default=None, description="self_ref of the containing table")
    table_row: int | None = Field(default=None, ge=0)
    table_col: int | None = Field(default=None, ge=0)

    @field_validator("cell_range")
    @classmethod
    def _check_cell_range(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.upper()
        if not _CELL_RANGE.match(v):
            raise ValueError(f"invalid cell range {v!r}; expected A1 or A1:C10")
        return v

    @model_validator(mode="after")
    def _check_format_fields(self) -> Self:
        if self.format is DocumentFormat.PDF and self.page_number is None:
            raise ValueError("PDF location requires page_number")
        if self.format is DocumentFormat.XLSX and not (self.sheet_name and self.cell_range):
            raise ValueError("XLSX location requires sheet_name and cell_range")
        if self.format is not DocumentFormat.XLSX and (self.sheet_name or self.cell_range):
            raise ValueError("sheet_name/cell_range only apply to XLSX")
        if self.format is not DocumentFormat.PDF and (self.page_number or self.page_label):
            raise ValueError("page_number/page_label only apply to PDF")
        if (self.table_row is None) != (self.table_col is None):
            raise ValueError("table_row and table_col must be given together")
        if self.table_row is not None and self.table_ref is None:
            raise ValueError("table coordinates require table_ref")
        return self


class DocumentSpan(_Strict):
    """One parsed element (heading, paragraph, table, table cell, ...)."""

    span_id: str = Field(min_length=1, description="'<document_id>:<element_id>[:r<row>c<col>]'")
    document_id: str
    document_version: int = Field(ge=1)
    kind: SpanKind
    text: str
    order: int = Field(ge=0, description="Reading order within the document")
    parent_span_id: str | None = None
    location: SourceLocation | None = None
    missing_provenance: str | None = Field(
        default=None, description="Why location is None. Required when it is."
    )

    @model_validator(mode="after")
    def _location_or_reason(self) -> Self:
        if (self.location is None) == (self.missing_provenance is None):
            raise ValueError("set exactly one of location or missing_provenance")
        return self


# ------------------------------------------------------- entities and claims


class Entity(_Strict):
    entity_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    normalized_name: str | None = Field(default=None, description="Set by entities.normalize")
    entity_type: EntityType | None = None
    aliases: list[str] = Field(default_factory=list)
    mention_ids: list[str] = Field(default_factory=list, description="Source references")


class EntityMention(_Strict):
    """One occurrence of an entity name in a located document span."""

    mention_id: str = Field(min_length=1)
    surface: str = Field(min_length=1, description="Exact text as it appears in the span")
    normalized: str = Field(min_length=1)
    entity_type: EntityType
    span_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    location: SourceLocation
    entity_id: str | None = Field(
        default=None, description="Assigned entity; None while the match is ambiguous"
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.char_end - self.char_start != len(self.surface):
            raise ValueError("offsets do not match surface length")
        return self


class EntityMatchCandidate(_Strict):
    """A possible identity between a mention and an entity, with the evidence for it."""

    mention_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    status: MatchStatus
    score: float = Field(ge=0.0, le=1.0)
    rule: str = Field(min_length=1, description="e.g. 'exact', 'alias_definition', 'token_set'")
    explanation: str = Field(min_length=1)


class EvidenceRole(StrEnum):
    """Why an evidence slice is attached to a claim."""

    QUOTE = "quote"  # text the model quoted, located in the source
    SCOPE = "scope"  # subject / text value / time found in the quote's sentence or row
    HEADER = "header"  # table column header supporting a currency, unit or boolean


class Evidence(_Strict):
    """A quoted, located piece of a document span that backs (or refutes) a claim."""

    evidence_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: int = Field(ge=1)
    quote: str = Field(min_length=1)
    char_start: int | None = Field(default=None, ge=0, description="Offset into the span text")
    char_end: int | None = Field(default=None, ge=0)
    location: SourceLocation
    role: EvidenceRole = EvidenceRole.QUOTE

    @model_validator(mode="after")
    def _check_offsets(self) -> Self:
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be given together")
        if self.char_start is not None and self.char_end is not None:
            if self.char_end <= self.char_start:
                raise ValueError("char_end must be greater than char_start")
            if self.char_end - self.char_start != len(self.quote):
                raise ValueError("offsets do not match quote length")
        return self

    @classmethod
    def from_span(
        cls,
        evidence_id: str,
        span: DocumentSpan,
        quote: str | None = None,
        role: EvidenceRole = EvidenceRole.QUOTE,
    ) -> Evidence:
        """Build evidence from a parsed span. Fails if the span is unlocated or the
        quote is not literally present in the span text."""
        if span.location is None:
            raise ValueError(f"span {span.span_id} has no location: {span.missing_provenance}")
        quote = span.text if quote is None else quote
        start = span.text.find(quote)
        if start < 0:
            raise ValueError(f"quote not found in span {span.span_id}")
        return cls(
            evidence_id=evidence_id,
            span_id=span.span_id,
            document_id=span.document_id,
            document_version=span.document_version,
            quote=quote,
            char_start=start,
            char_end=start + len(quote),
            location=span.location,
            role=role,
        )


class Quantity(_Strict):
    """Numeric value with a unit and/or ISO-4217 currency. Decimal avoids float drift."""

    value: Decimal
    unit: str | None = Field(default=None, description="e.g. 'percent', 'staff'; or legacy 'LKR'")
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$", description="ISO 4217")


ClaimObject = Quantity | bool | str


class MetadataField(StrEnum):
    CURRENCY = "currency"
    PERIOD = "period"


class MetadataOrigin(StrEnum):
    MODEL = "model"  # value supplied by the model and confirmed by the evidence
    ENRICHMENT = "enrichment"  # value read deterministically from the cited evidence
    NONE = "none"  # no value accepted


class MetadataStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"  # model value not supported by the evidence
    CONFLICT = "conflict"  # model value contradicts the evidence; neither is used
    AMBIGUOUS = "ambiguous"  # evidence allows several values; none is used
    ABSENT = "absent"  # nothing in the evidence
    NOT_APPLIED = "not_applied"  # evidence has a value the claim's type does not take


class MetadataDecision(_Strict):
    """Why a claim's currency or period is what it is (including null)."""

    field: MetadataField
    origin: MetadataOrigin
    status: MetadataStatus
    value: str | None = Field(default=None, description="Accepted value: 'LKR', '2025..2025'")
    model_value: str | None = Field(default=None, description="What the model said, verbatim")
    candidate_value: str | None = Field(default=None, description="What the evidence showed")
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        accepted = self.status is MetadataStatus.ACCEPTED
        if accepted != (self.value is not None):
            raise ValueError("value is set exactly when the status is accepted")
        if accepted and self.origin is MetadataOrigin.NONE:
            raise ValueError("an accepted value needs an origin")
        if not accepted and self.origin is not MetadataOrigin.NONE:
            raise ValueError("only accepted values have an origin")
        if accepted and not self.evidence_ids:
            raise ValueError("an accepted value must cite evidence")
        return self


class Claim(_Strict):
    claim_id: str = Field(min_length=1)
    subject: str = Field(min_length=1, description="Surface form as written in the source")
    subject_entity_id: str | None = Field(default=None, description="Set by entity resolution")
    predicate: str = Field(pattern=_SNAKE, description="snake_case, e.g. 'budget'")
    object: ClaimObject
    object_type: ObjectType | None = None
    object_entity_id: str | None = None
    modality: Modality = Modality.UNKNOWN
    valid_from: str | None = Field(default=None, description="YYYY, YYYY-MM or YYYY-MM-DD")
    valid_to: str | None = Field(default=None, description="YYYY, YYYY-MM or YYYY-MM-DD")
    time_expression: str | None = Field(
        default=None, description="Source text the period was read from, e.g. 'in January 2025'"
    )
    negated: bool = Field(default=False, description="True for 'X did not happen'")
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: Literal["llm", "rule", "table", "manual"] = "llm"
    evidence: list[Evidence] = Field(min_length=1)
    metadata: list[MetadataDecision] = Field(
        default_factory=list, description="Provenance of currency and period (enrichment)"
    )

    @field_validator("valid_from", "valid_to")
    @classmethod
    def _check_date(cls, v: str | None) -> str | None:
        if v is not None and parse_partial_date(v) is None:
            raise ValueError(
                f"invalid date {v!r}; expected a real calendar date as YYYY, YYYY-MM or YYYY-MM-DD"
            )
        return v

    @model_validator(mode="after")
    def _check_claim(self) -> Self:
        if self.valid_from and self.valid_to:
            # Compare at the coarser precision: "2025-06" .. "2025" is valid.
            n = min(len(self.valid_from), len(self.valid_to))
            if self.valid_to[:n] < self.valid_from[:n]:
                raise ValueError("valid_to is earlier than valid_from")
        ids = [e.evidence_id for e in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evidence_id in claim")
        cited = {i for d in self.metadata for i in d.evidence_ids}
        if cited - set(ids):
            raise ValueError("metadata decision cites evidence the claim does not have")
        if len({d.field for d in self.metadata}) != len(self.metadata):
            raise ValueError("at most one metadata decision per field")
        self._check_object_type()
        return self

    def _check_object_type(self) -> None:
        t, o = self.object_type, self.object
        if t is None:
            return
        if t is ObjectType.MONEY and not (isinstance(o, Quantity) and o.currency):
            raise ValueError("money object requires a Quantity with currency")
        if t is ObjectType.PERCENT and not (isinstance(o, Quantity) and o.unit == "percent"):
            raise ValueError("percent object requires a Quantity with unit 'percent'")
        if t is ObjectType.NUMBER and not isinstance(o, Quantity):
            raise ValueError("number object requires a Quantity")
        if t is ObjectType.DATE and not (isinstance(o, str) and parse_partial_date(o)):
            raise ValueError("date object must be YYYY, YYYY-MM or YYYY-MM-DD")
        if t is ObjectType.TEXT and not isinstance(o, str):
            raise ValueError("text object must be a string")
        if t is ObjectType.BOOLEAN and not isinstance(o, bool):
            raise ValueError("boolean object must be a bool")


class Relationship(_Strict):
    relationship_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    relation: RelationType
    target_id: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    reviewed_by_human: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.source_id == self.target_id:
            raise ValueError("relationship cannot point to itself")
        return self


# ------------------------------------------------------- conflicts and verdicts


class Conflict(_Strict):
    """Outcome of comparing two claims."""

    conflict_id: str = Field(min_length=1)
    claim_a_id: str = Field(min_length=1)
    claim_b_id: str = Field(min_length=1)
    verdict: Verdict
    conflict_type: ConflictType | None = None
    explanation: str = Field(min_length=1, description="Every conflict must be explained")
    detection_confidence: float = Field(ge=0.0, le=1.0)
    method: str = Field(min_length=1, description="e.g. 'rule:numeric', 'nli', 'llm:qwen3:4b'")
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.claim_a_id == self.claim_b_id:
            raise ValueError("a claim cannot conflict with itself")
        if self.verdict not in CONFLICT_VERDICTS:
            raise ValueError(f"{self.verdict} is not a claim-pair verdict")
        if (self.verdict is Verdict.CONTRADICTED) != (self.conflict_type is not None):
            raise ValueError("conflict_type is required for, and only for, CONTRADICTED")
        return self


class VerificationResult(_Strict):
    """Outcome of checking one claim against evidence."""

    verification_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    verdict: Verdict
    verification_confidence: float = Field(ge=0.0, le=1.0)
    method: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    refuting_evidence_ids: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def abstained(self) -> bool:
        return self.verdict in ABSTENTION_VERDICTS

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.verdict not in CLAIM_VERDICTS:
            raise ValueError(f"{self.verdict} is not a single-claim verdict")
        if self.verdict is Verdict.SUPPORTED and not self.supporting_evidence_ids:
            raise ValueError("SUPPORTED requires supporting evidence")
        if self.verdict is Verdict.REFUTED and not self.refuting_evidence_ids:
            raise ValueError("REFUTED requires refuting evidence")
        if self.abstained != (self.abstention_reason is not None):
            raise ValueError("abstention_reason is required for, and only for, abstentions")
        return self

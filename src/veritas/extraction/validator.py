"""Ground model candidates in source text. Model-independent and deterministic.

Evidence scope: a claim may rely only on its cited evidence, meaning the model's
quotes, the whole sentence(s) containing a quote (prose), or the cited cell's own
table row (tables). Headings, sheet names and other sentences of the same unit are
never evidence. When the subject, a text value or the time expression is found in
the scope but outside the quotes, its exact occurrence is added as evidence.

Core facts must be proven or the candidate is **rejected** (with a reason):
  source id, every evidence quote, subject, value (numbers must equal a number in
  the quotes; text values must be in scope), modality must not contradict the
  source wording, predicate wording, booleans (same clause / same row).
Optional context that cannot be proven is **removed** and recorded as an issue:
  currency (money becomes number), unit, time period, model-only modality.

Evidence text is always sliced from the parsed span, never copied from the model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from veritas.core.contracts import (
    Claim,
    ClaimObject,
    DocumentSpan,
    EntityMention,
    Evidence,
    EvidenceRole,
    MetadataDecision,
    MetadataField,
    MetadataOrigin,
    MetadataStatus,
    Modality,
    ObjectType,
    Quantity,
)
from veritas.entities.normalize import normalize_name
from veritas.extraction import enrichment as enrich
from veritas.extraction import grounding as g
from veritas.extraction.schema import LLMClaim, LLMEntity
from veritas.extraction.sources import SourceUnit

# Multiplicative confidence penalties per recorded issue (heuristic, uncalibrated).
PENALTIES: dict[str, float] = {
    "currency_not_grounded": 0.7,
    "time_not_grounded": 0.8,
    "modality_not_grounded": 0.8,
    "modality_ambiguous": 0.85,
    "unit_not_grounded": 0.9,
    "predicate_not_in_vocabulary": 0.95,
    "retried": 0.9,
    # enrichment (on) only:
    "currency_absent": 0.7,  # model said money, no currency in the evidence
    "currency_conflict": 0.6,
    "currency_ambiguous": 0.8,
    "period_precision_reduced": 0.95,
}


_NUMERIC = (ObjectType.MONEY, ObjectType.NUMBER, ObjectType.PERCENT)


# A supporting evidence slice: (span, start, end, role); a period candidate with its
# supporting slices.
_Slice = tuple[DocumentSpan, int, int, EvidenceRole]
_PeriodCandidate = tuple["enrich.Period", DocumentSpan, int, list[_Slice]]


@dataclass
class _PendingDecision:
    """A metadata decision before the claim's final evidence ids are known."""

    field: MetadataField
    origin: MetadataOrigin
    status: MetadataStatus
    value: str | None
    model_value: str | None
    candidate_value: str | None
    reason: str
    keys: list[tuple[str, int | None, int | None]]

    def to_decision(
        self, key_to_id: dict[tuple[str, int | None, int | None], str]
    ) -> MetadataDecision:
        return MetadataDecision(
            field=self.field,
            origin=self.origin,
            status=self.status,
            value=self.value,
            model_value=self.model_value,
            candidate_value=self.candidate_value,
            reason=self.reason,
            evidence_ids=[key_to_id[k] for k in self.keys if k in key_to_id],
        )


class Rejected(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class ExtractedClaim:
    claim: Claim
    source_id: str
    issues: list[str] = field(default_factory=list)
    fingerprint: str = ""


@dataclass
class Rejection:
    source_id: str
    kind: str  # "claim" | "entity"
    reason: str
    detail: str
    payload: dict[str, Any]


def _id(prefix: str, *parts: object) -> str:
    return f"{prefix}-{hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16]}"


class ClaimValidator:
    def __init__(
        self,
        units: list[SourceUnit],
        document_id: str,
        run_id: str,
        *,
        enrichment: bool = True,
        currency_defaults: Mapping[str, str] | None = None,
    ) -> None:
        self.units = {u.source_id: u for u in units}
        self.document_id = document_id
        self.run_id = run_id
        self.enrichment = enrichment
        self.currency_defaults: dict[str, str] = dict(currency_defaults or {})
        self._pending: list[_PendingDecision] = []

    # ------------------------------------------------------------------ claims

    def validate_claim(self, c: LLMClaim, attempt: int = 1) -> ExtractedClaim:
        unit = self.units.get(c.source_id)
        if unit is None:
            raise Rejected("unknown_source", f"source_id {c.source_id!r} was not provided")
        issues: list[str] = []
        self._pending = []

        evidence = self._evidence(c, unit)
        quotes_text = "\n".join(e.quote for e in evidence)  # the model's quotes only

        # The subject must be part of the cited evidence: in a quote, or in the same
        # sentence (prose) / same row (tables) as a quote, in which case its exact
        # occurrence is added as evidence. Elsewhere in the unit is not enough.
        if not self._ground(c.subject, evidence, unit):
            raise Rejected(
                "subject_not_grounded",
                f"{c.subject!r} is not in the cited evidence (its sentence or table row)",
            )

        predicate, implied, in_vocab = g.normalize_predicate(c.predicate)
        if not in_vocab:
            issues.append("predicate_not_in_vocabulary")
        modality = c.modality
        if implied is not None:
            if modality is Modality.UNKNOWN:
                modality = implied
            elif modality is not implied:
                raise Rejected(
                    "predicate_modality_conflict",
                    f"predicate {c.predicate!r} implies {implied}, modality is {modality}",
                )
        # Modality is read from the quotes; if they carry no modality word, from the
        # quotes' own sentences / table row. Never from headings or other sentences.
        modality = self._modality(c, modality, evidence, unit, issues)

        obj, object_type = self._object(c, predicate, unit, quotes_text, evidence, issues)
        valid_from, valid_to, time_expr = self._time(c, unit, evidence, issues)
        if self.enrichment:
            valid_from, valid_to, time_expr = self._enrich_period(
                c, object_type, obj, (valid_from, valid_to, time_expr), evidence, unit, issues
            )

        if attempt > 1:
            issues.append("retried")
        confidence = c.confidence
        for issue in issues:
            confidence *= PENALTIES.get(issue, 1.0)

        fingerprint = _id(
            "FP",
            self.document_id,
            normalize_name(c.subject),
            predicate,
            modality,
            object_type,
            obj,
            valid_from,
            valid_to,
            ",".join(sorted(e.span_id for e in evidence)),
        )
        claim_id = _id("CLM", self.run_id, fingerprint)
        # Evidence ids are per claim: two claims citing the same slice (a shared
        # sentence, subject cell or column header) each get their own record.
        evidence = [
            e.model_copy(
                update={"evidence_id": _id("EV", claim_id, e.span_id, e.char_start, e.char_end)}
            )
            for e in evidence
        ]
        key_to_id = {(e.span_id, e.char_start, e.char_end): e.evidence_id for e in evidence}
        metadata = [p.to_decision(key_to_id) for p in self._pending]
        claim = Claim(
            claim_id=claim_id,
            subject=c.subject,
            predicate=predicate,
            object=obj,
            object_type=object_type,
            modality=modality,
            valid_from=valid_from,
            valid_to=valid_to,
            time_expression=time_expr,
            negated=c.negated,
            extraction_confidence=round(min(max(confidence, 0.0), 1.0), 3),
            extraction_method="llm",
            evidence=evidence,
            metadata=metadata,
        )
        return ExtractedClaim(claim, c.source_id, issues, fingerprint)

    def _evidence(self, c: LLMClaim, unit: SourceUnit) -> list[Evidence]:
        out: dict[tuple[str, int], Evidence] = {}
        for quote in c.evidence_quotes:
            matches = self._find_all(quote, unit)
            if not matches:
                raise Rejected(
                    "quote_not_in_source", f"{quote!r} is not in source {unit.source_id}"
                )
            for span, (start, end) in matches:
                out[(span.span_id, start)] = self._make_evidence(span, start, end)
        return list(out.values())

    def _make_evidence(
        self, span: DocumentSpan, start: int, end: int, role: EvidenceRole = EvidenceRole.QUOTE
    ) -> Evidence:
        exact = span.text[start:end]  # the source's own characters
        try:
            ev = Evidence.from_span(
                _id("EV", self.run_id, span.span_id, start, end), span, exact, role
            )
        except ValueError as e:
            raise Rejected("missing_provenance", str(e)) from e
        if ev.char_start != start:  # an earlier identical substring; keep exact offsets
            ev = ev.model_copy(update={"char_start": start, "char_end": end})
        return ev

    # ------------------------------------------------------- evidence scope

    @staticmethod
    def _row_key(span: DocumentSpan) -> tuple[str, int] | None:
        loc = span.location
        if loc is None or loc.table_ref is None or loc.table_row is None:
            return None
        return loc.table_ref, loc.table_row

    def _row_cells(self, evidence: list[Evidence], unit: SourceUnit) -> list[DocumentSpan]:
        """Cells (in the unit) that share a table row with a cited cell."""
        by_id = {sp.span_id: sp for sp in unit.spans}
        keys = {self._row_key(by_id[e.span_id]) for e in evidence if e.span_id in by_id}
        keys.discard(None)
        return [sp for sp in unit.spans if self._row_key(sp) in keys]

    def _scopes(
        self, evidence: list[Evidence], unit: SourceUnit
    ) -> list[tuple[DocumentSpan, int, int]]:
        """(span, start, end) regions a claim's quotes legitimately cover: for prose,
        the whole sentence(s) containing each quote; for tables, every cell of the
        cited row(s)."""
        if unit.kind == "table_row":
            return [(sp, 0, len(sp.text)) for sp in self._row_cells(evidence, unit)]
        by_id = {sp.span_id: sp for sp in unit.spans}
        out = []
        for e in evidence:
            sp = by_id.get(e.span_id)
            if sp is None or e.char_start is None or e.char_end is None:
                continue
            s0, s1 = g.sentence_bounds(sp.text, e.char_start, e.char_end)
            out.append((sp, s0, s1))
        return out

    def _scope_text(self, evidence: list[Evidence], unit: SourceUnit) -> str:
        return "\n".join(sp.text[a:b] for sp, a, b in self._scopes(evidence, unit))

    def _ground(self, needle: str, evidence: list[Evidence], unit: SourceUnit) -> bool:
        """True if ``needle`` is in a quote, or in a quote's sentence / table row; in
        the latter case its exact occurrence is appended to ``evidence``."""
        if any(g.contains(needle, e.quote) for e in evidence):
            return True
        for sp, a, b in self._scopes(evidence, unit):
            pos = g.locate(needle, sp.text[a:b])
            if pos is not None:
                self._add(
                    evidence, self._make_evidence(sp, a + pos[0], a + pos[1], EvidenceRole.SCOPE)
                )
                return True
        return False

    @staticmethod
    def _add(evidence: list[Evidence], ev: Evidence) -> None:
        if all((x.span_id, x.char_start) != (ev.span_id, ev.char_start) for x in evidence):
            evidence.append(ev)

    def _headers_of(self, cells: list[DocumentSpan], unit: SourceUnit) -> list[DocumentSpan]:
        """Located header cells of the given data cells' columns."""
        if not unit.header_spans:
            return []
        by_id = dict(zip((sp.span_id for sp in unit.spans), unit.header_spans, strict=True))
        out = []
        for cell in cells:
            h = by_id.get(cell.span_id)
            if h is not None and h.location is not None and h not in out:
                out.append(h)
        return out

    def _header_support(
        self,
        evidence: list[Evidence],
        headers: list[DocumentSpan],
        find: Callable[[str], tuple[int, int] | None],
    ) -> bool:
        """If ``find`` locates the needed text in one of ``headers``, record that exact
        slice of the header cell as HEADER evidence and return True."""
        for h in headers:
            pos = find(h.text)
            if pos is not None:
                try:
                    ev = self._make_evidence(h, pos[0], pos[1], EvidenceRole.HEADER)
                except Rejected:
                    continue  # unlocated header: not traceable, so not support
                self._add(evidence, ev)
                return True
        return False

    def _value_cells(
        self, c: LLMClaim, evidence: list[Evidence], unit: SourceUnit
    ) -> list[DocumentSpan]:
        """Cited data cells holding the claim's value (all cited cells if non-numeric)."""
        cited = [sp for sp, _ in self._cited_cells(evidence, unit)]
        value = g.parse_value(c.value) if c.object_type in _NUMERIC else None
        if value is None:
            return cited
        return [sp for sp in cited if g.number_in(value, sp.text)]

    def _cited_cells(
        self, evidence: list[Evidence], unit: SourceUnit
    ) -> list[tuple[DocumentSpan, str | None]]:
        cited = {e.span_id for e in evidence}
        labels = unit.labels or tuple([None] * len(unit.spans))
        return [(sp, lb) for sp, lb in zip(unit.spans, labels, strict=True) if sp.span_id in cited]

    @classmethod
    def _find_all(cls, quote: str, unit: SourceUnit) -> list[tuple[DocumentSpan, tuple[int, int]]]:
        """Locate a quote. For table rows, a quote copied from the row rendering
        ('Header = value | Header = value') is split into pairs; each value must be
        in the cell under that header. Header text itself is never evidence."""
        span, pos = cls._find(quote, unit)
        if span is not None and pos is not None:
            return [(span, pos)]
        if unit.kind != "table_row" or not unit.labels:
            return []
        found: list[tuple[DocumentSpan, tuple[int, int]]] = []
        for part in (p.strip() for p in quote.split("|")):
            if not part:
                continue
            header, sep, value = part.partition("=")
            hit = None
            if sep:
                for cell, label in zip(unit.spans, unit.labels, strict=True):
                    if label is not None and g.locate(header.strip(), label) == (0, len(label)):
                        p = g.locate(value.strip(), cell.text)
                        if p is not None:
                            hit = (cell, p)
                        break
            else:
                s2, p2 = cls._find(part, unit)
                hit = (s2, p2) if s2 is not None and p2 is not None else None
            if hit is None:
                return []
            found.append(hit)
        return found

    @staticmethod
    def _find(quote: str, unit: SourceUnit) -> tuple[DocumentSpan | None, tuple[int, int] | None]:
        for span in unit.spans:
            pos = g.locate(quote, span.text)
            if pos is not None:
                return span, pos
        return None, None

    def _modality(
        self,
        c: LLMClaim,
        model: Modality,
        evidence: list[Evidence],
        unit: SourceUnit,
        issues: list[str],
    ) -> Modality:
        """Modality rule (documented in docs/data_model.md):

        The permitted scope is the WHOLE sentence(s) containing the quotes (prose) or
        the cited row plus the header of the value's column (tables); never other
        sentences, other columns' headers or headings. Let S be the modalities stated
        in that scope.
        * model modality not in S, S non-empty -> reject ``modality_conflict``
        * S empty                               -> ``unknown`` + ``modality_not_grounded``
        * S has several modalities              -> ``unknown`` + ``modality_ambiguous``
          ("Project Alpha was proposed, but later approved": quoting only the first
          half does not make the claim 'proposed')
        * S == {model}                          -> accepted
        """
        if model is Modality.UNKNOWN:
            return model
        scope = self._scope_text(evidence, unit)
        headers: list[DocumentSpan] = []
        if unit.kind == "table_row":
            headers = self._headers_of(self._value_cells(c, evidence, unit), unit)
            scope = "\n".join([scope, *(h.text for h in headers)])
        stated = g.modalities_in(scope)
        if model not in stated:
            if stated:
                raise Rejected(
                    "modality_conflict",
                    f"model said {model.value}; the claim's sentence/row states "
                    f"{sorted(m.value for m in stated)}",
                )
            issues.append("modality_not_grounded")
            return Modality.UNKNOWN
        if len(stated) > 1:
            issues.append("modality_ambiguous")
            return Modality.UNKNOWN
        if headers and model not in g.modalities_in(self._scope_text(evidence, unit)):
            # Stated only by the value's column header: cite that header.
            pattern = g.MODALITY_PATTERNS[model]
            self._header_support(
                evidence, headers, lambda t: m.span() if (m := pattern.search(t)) else None
            )
        return model

    def _object(
        self,
        c: LLMClaim,
        predicate: str,
        unit: SourceUnit,
        quotes: str,
        evidence: list[Evidence],
        issues: list[str],
    ) -> tuple[ClaimObject, ObjectType]:
        t = c.object_type
        if t in (ObjectType.MONEY, ObjectType.NUMBER, ObjectType.PERCENT):
            value = g.parse_value(c.value)
            if value is None:
                raise Rejected("invalid_value", f"{c.value!r} is not a number")
            if not g.number_in(value, quotes):
                raise Rejected("value_not_grounded", f"{value} does not appear in the quotes")
            if t is ObjectType.PERCENT:
                if "%" not in quotes and "percent" not in quotes.lower():
                    raise Rejected("value_not_grounded", "no percent sign or word in the quotes")
                return Quantity(value=value, unit="percent"), t
            # Units and currencies come only from the evidence: the quotes, or the
            # located header cell of the column holding the value ("Amount (LKR)"),
            # which is then recorded as HEADER evidence. Another sentence, a heading
            # or another column's header never counts.
            headers = self._headers_of(self._value_cells(c, evidence, unit), unit)
            if t is ObjectType.MONEY and self.enrichment:
                code_or_none = self._enrich_currency(c, value, evidence, unit, headers, issues)
                if code_or_none is not None:
                    return Quantity(value=value, currency=code_or_none), t
                return Quantity(value=value, unit=None), ObjectType.NUMBER
            if t is ObjectType.MONEY:
                code = (c.currency or "").strip().upper()
                if self._model_currency_supported(code, quotes, evidence, headers):
                    return Quantity(value=value, currency=code), t
                issues.append("currency_not_grounded")
                return Quantity(value=value, unit=None), ObjectType.NUMBER
            if self.enrichment and t is ObjectType.NUMBER:
                self._note_unapplied_currency(value, evidence, unit, headers)
            unit_name = None
            if c.unit:
                needle = c.unit
                if g.contains(needle, quotes) or self._header_support(
                    evidence, headers, lambda h: g.locate(needle, h)
                ):
                    unit_name = c.unit
                else:
                    issues.append("unit_not_grounded")
            return Quantity(value=value, unit=unit_name), t
        if t is ObjectType.DATE:
            if not g.is_partial_date(c.value):
                raise Rejected("invalid_value", f"{c.value!r} is not YYYY[-MM[-DD]]")
            if not g.date_in(c.value, quotes):
                raise Rejected("value_not_grounded", f"date {c.value} not stated in the quotes")
            return c.value, t
        if t is ObjectType.BOOLEAN:
            if c.value.lower() not in ("true", "false"):
                raise Rejected("invalid_value", f"{c.value!r} is not true/false")
            asserted = (c.value.lower() == "true") != c.negated
            polarity = self._boolean_polarity(c, predicate, quotes, evidence, unit)
            if polarity in ("absent", "ambiguous"):
                raise Rejected(
                    "boolean_not_grounded",
                    f"quotes do not {'clearly ' if polarity == 'ambiguous' else ''}state whether "
                    f"{c.subject!r} {predicate}",
                )
            if (polarity == "true") != asserted:
                raise Rejected(
                    "boolean_conflict", f"quotes state {polarity}, claim asserts {asserted}"
                )
            return c.value.lower() == "true", t
        if not self._ground(c.value, evidence, unit):
            raise Rejected(
                "value_not_grounded",
                f"{c.value!r} is not in the cited evidence (its sentence or table row)",
            )
        return c.value, t

    def _boolean_polarity(
        self,
        c: LLMClaim,
        predicate: str,
        quotes: str,
        evidence: list[Evidence],
        unit: SourceUnit,
    ) -> str:
        """Prose: a quoted clause with subject + predicate word (grounding.boolean_polarity).
        Tables: each cited row is judged on its own; only rows that contain the subject
        count, and the Yes/No cell must be in that same row under a header naming the
        predicate. Conflicting rows give 'ambiguous'."""
        cited = [(sp, lb) for sp, lb in self._cited_cells(evidence, unit) if lb]
        if not cited:
            return g.boolean_polarity(quotes, c.subject, predicate)
        labels = dict(zip((sp.span_id for sp in unit.spans), unit.labels, strict=True))
        rows: dict[tuple[str, int], list[DocumentSpan]] = {}
        for sp in unit.spans:
            key = self._row_key(sp)
            if key is not None:
                rows.setdefault(key, []).append(sp)
        found: set[str] = set()
        used: list[DocumentSpan] = []
        for sp, label in cited:
            key = self._row_key(sp)
            if key is None:
                continue
            row = rows.get(key, [])
            if not any(g.contains(c.subject, cell.text) for cell in row if cell is not sp):
                continue  # this cell's row is about another entity
            polarity = g.cell_polarity(label or labels.get(sp.span_id) or "", sp.text, predicate)
            if polarity != "absent":
                found.add(polarity)
                used.append(sp)
        if not found:
            return "absent"
        if len(found) > 1:
            return "ambiguous"
        # The header naming the predicate ("Completed") supports the Yes/No value.
        self._header_support(
            evidence,
            self._headers_of(used, unit),
            lambda h: (0, len(h)) if g.header_names_predicate(h, predicate) else None,
        )
        return found.pop()

    # --------------------------------------------------- currency (shared + enrichment)

    def _model_currency_supported(
        self, code: str, quotes: str, evidence: list[Evidence], headers: list[DocumentSpan]
    ) -> bool:
        """Original rule: the model's code is in the quotes, or in the value column's
        located header (recorded as HEADER evidence). Ambiguous symbols count only via
        an explicitly configured default."""
        d = self.currency_defaults
        return (
            len(code) == 3
            and code.isalpha()
            and (
                g.currency_in(code, quotes, d)
                or self._header_support(evidence, headers, lambda h: g.locate_currency(code, h, d))
            )
        )

    def _support(
        self, evidence: list[Evidence], span: DocumentSpan, start: int, end: int, role: EvidenceRole
    ) -> tuple[str, int | None, int | None]:
        """Key of the evidence entry that covers span[start:end]; adds one if needed."""
        for e in evidence:
            if (
                e.span_id == span.span_id
                and e.char_start is not None
                and e.char_end is not None
                and e.char_start <= start
                and end <= e.char_end
            ):
                return e.span_id, e.char_start, e.char_end
        ev = self._make_evidence(span, start, end, role)
        self._add(evidence, ev)
        return ev.span_id, ev.char_start, ev.char_end

    def _value_sites(
        self, value: Decimal, evidence: list[Evidence], unit: SourceUnit
    ) -> list[tuple[DocumentSpan, int, int, int, int]]:
        """(span, start, end, window_lo, window_hi) for each occurrence of the claim's
        value: inside a quote (window = the quote's sentence) or a cited value cell
        (window = the cell). Attachment is only searched within the window."""
        by_id = {sp.span_id: sp for sp in unit.spans}
        sites = []
        for e in evidence:
            sp = by_id.get(e.span_id)
            if (
                sp is None
                or e.char_start is None
                or e.char_end is None
                or e.role is not EvidenceRole.QUOTE
            ):
                continue
            if unit.kind == "table_row":
                lo, hi = 0, len(sp.text)
            else:
                lo, hi = g.sentence_bounds(sp.text, e.char_start, e.char_end)
            for a, b in enrich.value_occurrences(value, sp.text, e.char_start, e.char_end):
                sites.append((sp, a, b, lo, hi))
        return sites

    def _currency_finding(
        self,
        value: Decimal,
        evidence: list[Evidence],
        unit: SourceUnit,
        headers: list[DocumentSpan],
    ) -> tuple[dict[str, list[_Slice]], set[str]]:
        """Currencies attached to the value (code -> support slices) and unresolved
        ambiguous forms. The value column's own header counts; nothing else does."""
        found: dict[str, list[_Slice]] = {}
        ambiguous: set[str] = set()
        for sp, a, b, lo, hi in self._value_sites(value, evidence, unit):
            window = sp.text[lo:hi]
            for m in g.currencies_attached(window, a - lo, b - lo, self.currency_defaults):
                if m.code:
                    found.setdefault(m.code, []).append(
                        (sp, lo + m.start, lo + m.end, EvidenceRole.SCOPE)
                    )
                elif m.ambiguous_key:
                    ambiguous.add(m.surface)
        for h in headers:
            for m in g.currency_mentions(h.text, self.currency_defaults):
                if m.code:
                    found.setdefault(m.code, []).append((h, m.start, m.end, EvidenceRole.HEADER))
                elif m.ambiguous_key:
                    ambiguous.add(m.surface)
        return found, ambiguous

    def _enrich_currency(
        self,
        c: LLMClaim,
        value: Decimal,
        evidence: list[Evidence],
        unit: SourceUnit,
        headers: list[DocumentSpan],
        issues: list[str],
    ) -> str | None:
        """Currency for a MONEY claim, never overwriting the model silently:
        * model and evidence agree, or the model's code passes the original rule -> model
        * model silent, exactly one attached/header currency -> enrichment
        * model and evidence disagree -> neither (conflict, both recorded)
        * several currencies, or only an ambiguous symbol -> none (ambiguous)"""
        model = (c.currency or "").strip().upper() or None
        found, ambiguous = self._currency_finding(value, evidence, unit, headers)
        codes = set(found)
        quotes = "\n".join(e.quote for e in evidence if e.role is EvidenceRole.QUOTE)

        def accept(code: str, origin: MetadataOrigin, reason: str) -> str:
            keys = [
                self._support(evidence, sp, a, b, role)
                for sp, a, b, role in found.get(code, [])[:1]
            ]
            if not keys:  # model code accepted by the original rule (quote / header text)
                loc = next(
                    (
                        (e.span_id, e.char_start, e.char_end)
                        for e in evidence
                        if g.currency_in(code, e.quote, self.currency_defaults)
                    ),
                    None,
                )
                if loc is not None:
                    keys = [loc]
            self._pending.append(
                _PendingDecision(
                    MetadataField.CURRENCY,
                    origin,
                    MetadataStatus.ACCEPTED,
                    code,
                    model,
                    None,
                    reason,
                    keys,
                )
            )
            return code

        def reject(status: MetadataStatus, reason: str, issue: str, candidate: str | None) -> None:
            self._pending.append(
                _PendingDecision(
                    MetadataField.CURRENCY,
                    MetadataOrigin.NONE,
                    status,
                    None,
                    model,
                    candidate,
                    reason,
                    [],
                )
            )
            issues.append(issue)

        if len(codes) > 1:
            reject(
                MetadataStatus.AMBIGUOUS,
                "several_currencies_in_evidence",
                "currency_ambiguous",
                ",".join(sorted(codes)),
            )
            return None
        if model is not None:
            if codes and model not in codes:
                reject(
                    MetadataStatus.CONFLICT,
                    "model_currency_contradicts_evidence",
                    "currency_conflict",
                    next(iter(codes)),
                )
                return None
            if model in codes:
                return accept(model, MetadataOrigin.MODEL, "attached_to_value")
            if self._model_currency_supported(model, quotes, evidence, headers):
                return accept(model, MetadataOrigin.MODEL, "in_quote")
            reject(
                MetadataStatus.REJECTED,
                "model_currency_not_in_evidence",
                "currency_not_grounded",
                None,
            )
            return None
        if codes:
            (code,) = codes
            role = found[code][0][3]
            return accept(
                code,
                MetadataOrigin.ENRICHMENT,
                "value_column_header" if role is EvidenceRole.HEADER else "attached_to_value",
            )
        if ambiguous:
            reject(
                MetadataStatus.AMBIGUOUS,
                "ambiguous_currency_symbol",
                "currency_ambiguous",
                ",".join(sorted(ambiguous)),
            )
            return None
        reject(MetadataStatus.ABSENT, "no_currency_in_evidence", "currency_absent", None)
        return None

    def _note_unapplied_currency(
        self,
        value: Decimal,
        evidence: list[Evidence],
        unit: SourceUnit,
        headers: list[DocumentSpan],
    ) -> None:
        """A NUMBER claim whose value has an attached currency: the type is the model's
        decision, so nothing changes; the finding is recorded."""
        found, _ = self._currency_finding(value, evidence, unit, headers)
        if found:
            self._pending.append(
                _PendingDecision(
                    MetadataField.CURRENCY,
                    MetadataOrigin.NONE,
                    MetadataStatus.NOT_APPLIED,
                    None,
                    None,
                    ",".join(sorted(found)),
                    "claim_is_not_money",
                    [],
                )
            )

    # ----------------------------------------------------------------- period

    def _period_candidates(
        self, evidence: list[Evidence], unit: SourceUnit, exclude: list[tuple[str, int, int]]
    ) -> tuple[
        list[tuple[enrich.Period, DocumentSpan, int, list[_Slice]]],
        list[str],
    ]:
        """Periods stated in the claim's evidence scope, preferring the quotes. Each
        candidate carries its support slices. Returns (candidates, notes)."""
        notes: list[str] = []
        out: list[tuple[enrich.Period, DocumentSpan, int, list[_Slice]]] = []
        if unit.kind == "table_row":
            cells = self._row_cells(evidence, unit)
            hdr = dict(
                zip(
                    (sp.span_id for sp in unit.spans),
                    unit.header_spans or [None] * len(unit.spans),
                    strict=True,
                )
            )
            roles: dict[str, list[tuple[DocumentSpan, DocumentSpan, tuple[str, str, bool]]]] = {}
            unlabelled = False
            for cell in cells:
                parsed = enrich.parse_date_expression(cell.text)
                if not isinstance(parsed, tuple):
                    if parsed == enrich.INVALID:
                        notes.append("invalid_date")
                    continue
                h = hdr.get(cell.span_id)
                role = (
                    enrich.header_time_role(h.text)
                    if h is not None and h.location is not None
                    else None
                )
                if role is None:
                    unlabelled = True
                    continue
                roles.setdefault(role, []).append((cell, h, parsed))  # type: ignore[arg-type]
            if unlabelled and not roles:
                notes.append("row_date_unlabelled")

            def support(cell: DocumentSpan, h: DocumentSpan) -> list[_Slice]:
                return [
                    (cell, 0, len(cell.text), EvidenceRole.SCOPE),
                    (h, 0, len(h.text), EvidenceRole.HEADER),
                ]

            if (
                "start" in roles
                and "end" in roles
                and len(roles["start"]) == len(roles["end"]) == 1
            ):
                (sc, sh, sp_), (ec, eh, ep) = roles["start"][0], roles["end"][0]
                if sp_[0] <= ep[1]:
                    p = enrich.Period(sp_[0], ep[1], 0, len(sc.text), f"{sc.text}..{ec.text}")
                    out.append((p, sc, 0, support(sc, sh) + support(ec, eh)))
            for cell, h, (vf, vt, reduced) in roles.get("period", []):
                p = enrich.Period(vf, vt, 0, len(cell.text), cell.text, reduced)
                out.append((p, cell, 0, support(cell, h)))
            return out, notes

        by_id = {sp.span_id: sp for sp in unit.spans}
        in_quote: list[_PeriodCandidate] = []
        in_sentence: list[_PeriodCandidate] = []
        for e in evidence:
            sp = by_id.get(e.span_id)
            if (
                sp is None
                or e.char_start is None
                or e.char_end is None
                or e.role is not EvidenceRole.QUOTE
            ):
                continue
            lo, hi = g.sentence_bounds(sp.text, e.char_start, e.char_end)
            scan = enrich.find_periods(sp.text[lo:hi])
            notes += [f"invalid_date:{x}" for x in scan.invalid]
            notes += [f"unsupported:{x}" for x in scan.unsupported]
            for p in scan.periods:
                a, b = lo + p.start, lo + p.end
                if any(sp.span_id == sid and a < xe and b > xs for sid, xs, xe in exclude):
                    continue  # this date is the claim's own value
                item = (p, sp, lo, [(sp, a, b, EvidenceRole.SCOPE)])
                (in_quote if e.char_start <= a and b <= e.char_end else in_sentence).append(item)
        return (in_quote or in_sentence), notes

    def _enrich_period(
        self,
        c: LLMClaim,
        object_type: ObjectType,
        obj: ClaimObject,
        model_result: tuple[str | None, str | None, str | None],
        evidence: list[Evidence],
        unit: SourceUnit,
        issues: list[str],
    ) -> tuple[str | None, str | None, str | None]:
        """Period, never overwriting the model silently:
        * model period passed the original check -> kept (origin model)
        * model gave a period the evidence does not support -> nothing substituted;
          conflict recorded if the evidence states a different single period
        * model silent -> exactly one stated period in scope is used (enrichment);
          several -> ambiguous; none -> absent"""
        vf, vt, expr = model_result
        model_said = bool(c.valid_from or c.valid_to or c.time_expression)
        model_label = (
            f"{c.valid_from}..{c.valid_to}" if (c.valid_from or c.valid_to) else c.time_expression
        )

        def pend(
            origin: MetadataOrigin,
            status: MetadataStatus,
            value: str | None,
            cand: str | None,
            reason: str,
            keys: list[tuple[str, int | None, int | None]],
        ) -> None:
            self._pending.append(
                _PendingDecision(
                    MetadataField.PERIOD,
                    origin,
                    status,
                    value,
                    model_label if model_said else None,
                    cand,
                    reason,
                    keys,
                )
            )

        if vf is not None or vt is not None:  # model period accepted by the original rule
            key = next(
                (
                    (e.span_id, e.char_start, e.char_end)
                    for e in evidence
                    if expr and g.contains(expr, e.quote)
                ),
                None,
            )
            pend(
                MetadataOrigin.MODEL,
                MetadataStatus.ACCEPTED,
                f"{vf}..{vt}",
                None,
                "model_period_in_evidence",
                [key] if key else [],
            )
            return vf, vt, expr

        exclude: list[tuple[str, int, int]] = []
        if object_type is ObjectType.DATE:
            exclude = [
                (e.span_id, e.char_start or 0, e.char_end or 0)
                for e in evidence
                if e.role is EvidenceRole.QUOTE and isinstance(obj, str) and g.date_in(obj, e.quote)
            ]
        elif isinstance(obj, Quantity):
            for sp, a, b, _, _ in self._value_sites(obj.value, evidence, unit):
                exclude.append((sp.span_id, a, b))
        candidates, notes = self._period_candidates(evidence, unit, exclude)
        distinct = {p.key for p, _, _, _ in candidates}

        if model_said:  # the model's period failed the original rule (issue already recorded)
            cand = (
                next(iter({p.label for p, _, _, _ in candidates})) if len(distinct) == 1 else None
            )
            if cand and cand != model_label:
                pend(
                    MetadataOrigin.NONE,
                    MetadataStatus.CONFLICT,
                    None,
                    cand,
                    "model_period_contradicts_evidence",
                    [],
                )
            else:
                pend(
                    MetadataOrigin.NONE,
                    MetadataStatus.REJECTED,
                    None,
                    cand,
                    "model_period_not_in_evidence",
                    [],
                )
            return None, None, None
        if len(distinct) > 1:
            pend(
                MetadataOrigin.NONE,
                MetadataStatus.AMBIGUOUS,
                None,
                ",".join(sorted(f"{a}..{b}" for a, b in distinct)),
                "several_periods_in_evidence",
                [],
            )
            issues.append("period_ambiguous")
            return None, None, None
        if not candidates:
            reason = (
                "invalid_date_in_evidence"
                if any(n.startswith("invalid") for n in notes)
                else "unsupported_time_expression"
                if any(n.startswith("unsupported") for n in notes)
                else "row_date_unlabelled"
                if "row_date_unlabelled" in notes
                else "no_period_in_evidence"
            )
            pend(MetadataOrigin.NONE, MetadataStatus.ABSENT, None, None, reason, [])
            return None, None, None
        p, _, _, support = candidates[0]
        keys = [self._support(evidence, sp, a, b, role) for sp, a, b, role in support]
        reason = "stated_in_evidence"
        if p.precision_reduced:
            reason = "precision_reduced_to_year"
            issues.append("period_precision_reduced")
        pend(MetadataOrigin.ENRICHMENT, MetadataStatus.ACCEPTED, p.label, None, reason, keys)
        return p.valid_from, p.valid_to, p.expression

    def _time(
        self, c: LLMClaim, unit: SourceUnit, evidence: list[Evidence], issues: list[str]
    ) -> tuple[str | None, str | None, str | None]:
        """The period must come from the cited evidence: the time expression has to be
        in a quote, or in a quote's sentence / table row (then added as evidence), and
        must itself state every date at its precision. Headings, sheet names and other
        sentences of the unit never supply a period."""
        if not (c.valid_from or c.valid_to or c.time_expression):
            return None, None, None
        expr = c.time_expression or ""
        dates = [d for d in (c.valid_from, c.valid_to) if d]
        if (
            not expr
            or not all(g.is_partial_date(d) and g.date_in(d, expr) for d in dates)
            or not self._ground(expr, evidence, unit)
        ):
            issues.append("time_not_grounded")
            return None, None, None
        if c.valid_from and c.valid_to:
            n = min(len(c.valid_from), len(c.valid_to))
            if c.valid_to[:n] < c.valid_from[:n]:
                issues.append("time_not_grounded")
                return None, None, None
        return c.valid_from, c.valid_to, c.time_expression

    # ---------------------------------------------------------------- entities

    def validate_entity(self, e: LLMEntity) -> tuple[list[EntityMention], list[str]]:
        """Mentions for the entity's name (and any alias found in the same unit).
        Returns (mentions, aliases_not_found)."""
        unit = self.units.get(e.source_id)
        if unit is None:
            raise Rejected("unknown_source", f"source_id {e.source_id!r} was not provided")
        span, pos = self._find(e.name, unit)
        if span is None or pos is None:
            raise Rejected("entity_not_in_source", f"{e.name!r} is not in source {e.source_id}")
        mentions = [self._mention(span, pos, e)]
        missing: list[str] = []
        for alias in e.aliases:
            a_span, a_pos = self._find(alias, unit)
            if a_span is None or a_pos is None:
                a_span, a_pos = next(
                    (
                        (s, p)
                        for u in self.units.values()
                        for s in u.spans
                        if (p := g.locate(alias, s.text)) is not None
                    ),
                    (None, None),
                )
            if a_span is None or a_pos is None:
                missing.append(alias)
            else:
                mentions.append(self._mention(a_span, a_pos, e))
        return mentions, missing

    def _mention(self, span: DocumentSpan, pos: tuple[int, int], e: LLMEntity) -> EntityMention:
        if span.location is None:
            raise Rejected("missing_provenance", f"span {span.span_id} has no location")
        start, end = pos
        surface = span.text[start:end]
        return EntityMention(
            mention_id=_id("MEN", self.run_id, span.span_id, start, end, e.entity_type),
            surface=surface,
            normalized=normalize_name(surface),
            entity_type=e.entity_type,
            span_id=span.span_id,
            document_id=span.document_id,
            char_start=start,
            char_end=end,
            location=span.location,
        )


def decimal_str(d: Decimal) -> str:
    return format(d.normalize(), "f")

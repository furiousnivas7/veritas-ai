"""Entity resolution: assign mentions to entities without over-merging.

Identity model
--------------
Names do not identify real-world entities: two organisations, or two projects
in different documents, can share a name. Entity ids are therefore minted per
entity (derived from the mention that founded it), never from the name, and the
database allows any number of entities with the same type and normalised name.

Merge rules (MERGED), only within the **same entity type** and only when the
evidence is local enough to be trusted:

1. ``exact_same_document``: the normalised name or a proven alias equals one
   entity that already has a mention **in the same document** (a document is
   assumed to use one name for one entity).
2. ``alias_definition``: the document itself defines one name as the acronym of
   the other ("University of Sri Jayewardenepura (USJ)") and the other name's
   entity is in the same document.

Review rules (NEEDS_REVIEW, never merged). The mention starts its own entity and
a candidate is recorded:

* ``exact_cross_document``: same name as an entity seen only in other documents
* ``alias_definition_cross_document``: this document's alias definition points
  at an entity seen only in other documents
* ``ambiguous_exact`` / ``ambiguous_alias_definition``: the name, or this
  document's alias definition, matches several entities in the same document
  (the mention is left unassigned)
* ``token_set``, ``acronym``, ``fuzzy``, ``model_alias``: weaker resemblances

Policy ``global_name_match`` restores name-only merging across documents. It
exists as the explicit "exact entity matching" baseline for experiment E5 and is
not the default.

Different entity types never merge.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

from veritas.core.contracts import (
    Entity,
    EntityMatchCandidate,
    EntityMention,
    EntityType,
    MatchStatus,
)
from veritas.entities.normalize import (
    acronym,
    looks_like_acronym,
    normalize_name,
    significant_tokens,
)

FUZZY_THRESHOLD = 0.88
ResolutionPolicy = Literal["document_scoped", "global_name_match"]


def entity_id_for_mention(mention_id: str) -> str:
    """Id for an entity founded by ``mention_id``. Unique because mention ids are
    unique (they include the run, span and offsets); stable because it is stored
    once and never recomputed from the entity's name."""
    return f"ENT-{hashlib.sha256(mention_id.encode()).hexdigest()[:16]}"


@dataclass
class ResolutionResult:
    entities: dict[str, Entity] = field(default_factory=dict)  # new or updated entities
    mentions: list[EntityMention] = field(default_factory=list)
    candidates: list[EntityMatchCandidate] = field(default_factory=list)


class EntityResolver:
    def __init__(
        self,
        existing: Iterable[Entity] = (),
        entity_documents: Mapping[str, Iterable[str]] | None = None,
        policy: ResolutionPolicy = "document_scoped",
    ) -> None:
        """``entity_documents`` maps entity id -> ids of documents it has mentions in
        (from the database). Entities without an entry count as seen elsewhere."""
        self._entities: dict[str, Entity] = {e.entity_id: e for e in existing}
        self._docs: dict[str, set[str]] = {
            eid: set(docs) for eid, docs in (entity_documents or {}).items()
        }
        self._policy = policy
        self._touched: set[str] = set()

    def resolve(
        self,
        mentions: Iterable[EntityMention],
        alias_definitions: Iterable[tuple[str, str]] = (),
        model_aliases: Iterable[tuple[str, str]] = (),
    ) -> ResolutionResult:
        """``alias_definitions``: (long, short) pairs verified in the text of the
        document(s) being resolved. ``model_aliases``: (name, alias) pairs suggested
        by the LLM (review only)."""
        definitions = {
            frozenset({normalize_name(long), normalize_name(short)})
            for long, short in alias_definitions
        }
        result = ResolutionResult()
        unique = {m.mention_id: m for m in mentions}.values()  # a mention is resolved once
        ordered = sorted(unique, key=lambda m: (m.document_id, m.span_id, m.char_start, m.char_end))
        for mention in ordered:
            assigned, cands = self._resolve_one(mention, definitions)
            result.mentions.append(mention.model_copy(update={"entity_id": assigned}))
            result.candidates.extend(cands)

        result.candidates.extend(self._model_alias_candidates(model_aliases, result.mentions))
        result.entities = {eid: self._entities[eid] for eid in sorted(self._touched)}
        result.candidates = _dedupe(result.candidates)
        return result

    # ------------------------------------------------------------------ internals

    def _same_type(self, t: EntityType) -> list[Entity]:
        return [e for e in self._entities.values() if e.entity_type is t]

    def _keys(self, e: Entity) -> set[str]:
        return {e.normalized_name or normalize_name(e.canonical_name)} | {
            normalize_name(a) for a in e.aliases
        }

    def _in_document(self, e: Entity, document_id: str) -> bool:
        return document_id in self._docs.get(e.entity_id, set())

    def _resolve_one(
        self, m: EntityMention, definitions: set[frozenset[str]]
    ) -> tuple[str | None, list[EntityMatchCandidate]]:
        same_type = self._same_type(m.entity_type)
        global_policy = self._policy == "global_name_match"

        # 1. Exact name / proven alias.
        exact = [e for e in same_type if m.normalized in self._keys(e)]
        local = (
            exact if global_policy else [e for e in exact if self._in_document(e, m.document_id)]
        )
        if len(local) == 1:
            rule = "exact" if global_policy else "exact_same_document"
            self._attach(local[0], m)
            return local[0].entity_id, [
                _cand(m, local[0], MatchStatus.MERGED, 1.0, rule, "normalised name matches")
            ]
        if len(local) > 1:
            return None, [
                _cand(
                    m,
                    e,
                    MatchStatus.NEEDS_REVIEW,
                    0.9,
                    "ambiguous_exact",
                    f"name matches {len(local)} {m.entity_type.value} entities",
                )
                for e in local
            ]

        # 2. Alias defined in this document's text.
        defined = [
            e
            for e in same_type
            if any(frozenset({m.normalized, k}) in definitions for k in self._keys(e))
        ]
        defined_local = (
            defined
            if global_policy
            else [e for e in defined if self._in_document(e, m.document_id)]
        )
        if len(defined_local) == 1:
            self._attach(defined_local[0], m, alias=m.surface)
            return defined_local[0].entity_id, [
                _cand(
                    m,
                    defined_local[0],
                    MatchStatus.MERGED,
                    0.95,
                    "alias_definition",
                    "the document defines one name as the acronym of the other",
                )
            ]
        if len(defined_local) > 1:
            return None, [
                _cand(
                    m,
                    e,
                    MatchStatus.NEEDS_REVIEW,
                    0.85,
                    "ambiguous_alias_definition",
                    f"the alias definition matches {len(defined_local)} "
                    f"{m.entity_type.value} entities",
                )
                for e in defined_local
            ]

        # 3. No local proof: the mention founds its own entity; resemblances to
        #    entities elsewhere are queued for review.
        new = self._create(m)
        cands = [
            _cand(
                m,
                e,
                MatchStatus.NEEDS_REVIEW,
                0.9,
                "exact_cross_document",
                f"same name as {e.canonical_name!r}, which is only mentioned in other documents",
            )
            for e in exact
        ]
        cands += [
            _cand(
                m,
                e,
                MatchStatus.NEEDS_REVIEW,
                0.85,
                "alias_definition_cross_document",
                f"this document defines an alias of {e.canonical_name!r}, an entity from "
                "other documents",
            )
            for e in defined
            if e not in exact
        ]
        others = [e for e in same_type if e.entity_id != new.entity_id]
        return new.entity_id, cands + self._review_candidates(m, others)

    def _review_candidates(
        self, m: EntityMention, others: list[Entity]
    ) -> list[EntityMatchCandidate]:
        out: list[EntityMatchCandidate] = []
        m_tokens = significant_tokens(m.surface)
        for e in others:
            name = e.canonical_name
            if normalize_name(name) == m.normalized:
                continue  # handled as exact_* above
            if m_tokens and m_tokens == significant_tokens(name):
                out.append(
                    _cand(
                        m,
                        e,
                        MatchStatus.NEEDS_REVIEW,
                        0.8,
                        "token_set",
                        f"same words as {name!r} in a different order",
                    )
                )
            elif (
                looks_like_acronym(m.surface)
                and m.surface.replace(".", "").upper() == acronym(name)
            ) or (looks_like_acronym(name) and name.replace(".", "").upper() == acronym(m.surface)):
                out.append(
                    _cand(
                        m,
                        e,
                        MatchStatus.NEEDS_REVIEW,
                        0.6,
                        "acronym",
                        f"initials match {name!r} but no definition was found",
                    )
                )
            else:
                ratio = SequenceMatcher(None, m.normalized, normalize_name(name)).ratio()
                if ratio >= FUZZY_THRESHOLD:
                    out.append(
                        _cand(
                            m,
                            e,
                            MatchStatus.NEEDS_REVIEW,
                            round(ratio, 3),
                            "fuzzy",
                            f"spelling similar to {name!r}",
                        )
                    )
        return out

    def _model_alias_candidates(
        self, pairs: Iterable[tuple[str, str]], mentions: list[EntityMention]
    ) -> list[EntityMatchCandidate]:
        by_norm: dict[str, list[EntityMention]] = {}
        for mm in mentions:
            if mm.entity_id:
                by_norm.setdefault(mm.normalized, []).append(mm)
        out: list[EntityMatchCandidate] = []
        for name, alias in pairs:
            for a in by_norm.get(normalize_name(alias), []):
                for n in by_norm.get(normalize_name(name), []):
                    if a.entity_id and n.entity_id and a.entity_id != n.entity_id:
                        out.append(
                            _cand(
                                a,
                                self._entities[n.entity_id],
                                MatchStatus.NEEDS_REVIEW,
                                0.5,
                                "model_alias",
                                "the model suggested this alias; the text does not define it",
                            )
                        )
        return out

    def _create(self, m: EntityMention) -> Entity:
        e = Entity(
            entity_id=entity_id_for_mention(m.mention_id),
            canonical_name=m.surface,
            normalized_name=m.normalized,
            entity_type=m.entity_type,
        )
        self._entities[e.entity_id] = e
        return self._attach(e, m)

    def _attach(self, e: Entity, m: EntityMention, alias: str | None = None) -> Entity:
        aliases = list(e.aliases)
        canonical = e.canonical_name
        if alias and normalize_name(alias) not in self._keys(e):
            if looks_like_acronym(canonical) and not looks_like_acronym(alias):
                # Prefer the defined long form as the display name; keep the acronym.
                aliases.append(canonical)
                canonical = alias
            else:
                aliases.append(alias)
        updated = e.model_copy(
            update={
                "canonical_name": canonical,
                "aliases": aliases,
                "mention_ids": [*e.mention_ids, m.mention_id]
                if m.mention_id not in e.mention_ids
                else e.mention_ids,
            }
        )
        self._entities[e.entity_id] = updated
        self._docs.setdefault(e.entity_id, set()).add(m.document_id)
        self._touched.add(e.entity_id)
        return updated


def _cand(
    m: EntityMention, e: Entity, status: MatchStatus, score: float, rule: str, explanation: str
) -> EntityMatchCandidate:
    return EntityMatchCandidate(
        mention_id=m.mention_id,
        entity_id=e.entity_id,
        status=status,
        score=score,
        rule=rule,
        explanation=explanation,
    )


def _dedupe(cands: list[EntityMatchCandidate]) -> list[EntityMatchCandidate]:
    seen: dict[tuple[str, str], EntityMatchCandidate] = {}
    for c in cands:
        key = (c.mention_id, c.entity_id)
        if key not in seen or c.score > seen[key].score:
            seen[key] = c
    return list(seen.values())

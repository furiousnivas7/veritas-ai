from __future__ import annotations

import pytest

from veritas.core.contracts import (
    DocumentFormat,
    Entity,
    EntityMention,
    EntityType,
    MatchStatus,
    SourceLocation,
)
from veritas.entities.normalize import (
    acronym,
    find_alias_definitions,
    looks_like_acronym,
    normalize_name,
)
from veritas.entities.resolver import EntityResolver, entity_id_for_mention

ORG = EntityType.ORGANIZATION
LOC = SourceLocation(format=DocumentFormat.DOCX, element_id="#/texts/1")


def mention(surface: str, etype: EntityType = ORG, n: int = 0, doc: str = "DOC-1") -> EntityMention:
    return EntityMention(
        mention_id=f"MEN-{doc}-{n}",
        surface=surface,
        normalized=normalize_name(surface),
        entity_type=etype,
        span_id=f"{doc}:#/texts/{n}",
        document_id=doc,
        char_start=0,
        char_end=len(surface),
        location=LOC,
    )


# ----------------------------------------------------------------- normalize


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("The University of Sri Jayewardenepura.", "university of sri jayewardenepura"),
        ("PROJECT  Alpha", "project alpha"),
        ("R&D Unit", "r and d unit"),
        ("USJ", "usj"),
    ],
)
def test_normalize_name(raw: str, norm: str) -> None:
    assert normalize_name(raw) == norm


def test_acronym() -> None:
    assert acronym("University of Sri Jayewardenepura") == "USJ"
    assert looks_like_acronym("USJ") and looks_like_acronym("U.S.J.")
    assert not looks_like_acronym("Project Alpha")


def test_alias_definition_takes_matching_suffix() -> None:
    text = "the Research Office of the University of Sri Jayewardenepura (USJ) during 2025."
    (d,) = find_alias_definitions(text)
    assert (d.long_form, d.short_form) == ("University of Sri Jayewardenepura", "USJ")
    assert text[d.long_start : d.long_start + len(d.long_form)] == d.long_form


@pytest.mark.parametrize(
    "text", ["Project Alpha (2025) was approved.", "a budget of 5 million (LKR)", "Alpha (Beta)"]
)
def test_parentheses_that_are_not_acronym_definitions(text: str) -> None:
    assert find_alias_definitions(text) == []


# ------------------------------------------------------------------ resolver


def test_exact_names_merge_within_one_document() -> None:
    first = mention("Project Alpha", EntityType.PROJECT, 0)
    r = EntityResolver().resolve([first, mention("PROJECT ALPHA", EntityType.PROJECT, 1)])
    assert len(r.entities) == 1
    # The id comes from the founding mention, not from the name.
    assert {m.entity_id for m in r.mentions} == {entity_id_for_mention(first.mention_id)}
    assert [c.rule for c in r.candidates] == ["exact_same_document"]


def test_defined_acronym_merges_and_keeps_long_form_as_canonical() -> None:
    long = "University of Sri Jayewardenepura"
    for order in (
        [mention("USJ", n=0), mention(long, n=1)],
        [mention(long, n=0), mention("USJ", n=1)],
    ):
        r = EntityResolver().resolve(order, alias_definitions=[(long, "USJ")])
        (e,) = r.entities.values()
        assert e.canonical_name == long
        assert "USJ" in e.aliases or normalize_name("USJ") == e.normalized_name
        assert len({m.entity_id for m in r.mentions}) == 1
        assert any(
            c.rule == "alias_definition" and c.status is MatchStatus.MERGED for c in r.candidates
        )


def test_reordered_name_is_review_candidate_not_merge() -> None:
    r = EntityResolver().resolve(
        [
            mention("University of Sri Jayewardenepura", n=0),
            mention("Sri Jayewardenepura University", n=1),
        ]
    )
    assert len(r.entities) == 2
    (c,) = [c for c in r.candidates if c.status is MatchStatus.NEEDS_REVIEW]
    assert c.rule == "token_set"


def test_undefined_acronym_is_review_candidate() -> None:
    r = EntityResolver().resolve(
        [mention("University of Sri Jayewardenepura", n=0), mention("USJ", n=1)]
    )
    assert len(r.entities) == 2
    assert [c.rule for c in r.candidates if c.status is MatchStatus.NEEDS_REVIEW] == ["acronym"]


def test_similar_but_different_projects_are_not_linked() -> None:
    r = EntityResolver().resolve(
        [
            mention("Project Alpha", EntityType.PROJECT, 0),
            mention("Project Beta", EntityType.PROJECT, 1),
        ]
    )
    assert len(r.entities) == 2
    assert not [c for c in r.candidates if c.status is not MatchStatus.MERGED]


def test_typo_is_fuzzy_review_candidate() -> None:
    r = EntityResolver().resolve(
        [
            mention("University of Sri Jayewardenepura", n=0),
            mention("University of Sri Jayawardenepura", n=1),
        ]
    )
    assert len(r.entities) == 2
    assert [c.rule for c in r.candidates if c.status is MatchStatus.NEEDS_REVIEW] == ["fuzzy"]


def test_different_types_never_merge() -> None:
    r = EntityResolver().resolve(
        [mention("Alpha", EntityType.PROJECT, 0), mention("Alpha", EntityType.ORGANIZATION, 1)],
        alias_definitions=[("Alpha", "Alpha")],
    )
    assert len(r.entities) == 2


def test_ambiguous_exact_match_is_not_assigned() -> None:
    a = Entity(
        entity_id="ENT-a",
        canonical_name="University of Sri Jayewardenepura",
        normalized_name="university of sri jayewardenepura",
        entity_type=ORG,
        aliases=["USJ"],
    )
    b = Entity(
        entity_id="ENT-b",
        canonical_name="University of Sri Jayawardena Studies",
        normalized_name="university of sri jayawardena studies",
        entity_type=ORG,
        aliases=["USJ"],
    )
    both_here = {"ENT-a": {"DOC-1"}, "ENT-b": {"DOC-1"}}
    r = EntityResolver([a, b], both_here).resolve([mention("USJ")])
    (m,) = r.mentions
    assert m.entity_id is None
    assert {c.entity_id for c in r.candidates} == {"ENT-a", "ENT-b"}
    assert all(c.status is MatchStatus.NEEDS_REVIEW for c in r.candidates)
    assert {c.rule for c in r.candidates} == {"ambiguous_exact"}


USJ_EXISTING = Entity(
    entity_id="ENT-usj",
    canonical_name="University of Sri Jayewardenepura",
    normalized_name="university of sri jayewardenepura",
    entity_type=ORG,
    aliases=["USJ"],
)


def test_known_alias_in_another_document_is_review_not_merge() -> None:
    """Name equality across documents is not proof of identity (default policy)."""
    r = EntityResolver([USJ_EXISTING], {"ENT-usj": {"DOC-1"}}).resolve(
        [mention("USJ", doc="DOC-2")]
    )
    (m,) = r.mentions
    assert m.entity_id not in (None, "ENT-usj")  # founds its own entity
    (c,) = [c for c in r.candidates if c.entity_id == "ENT-usj"]
    assert (c.status, c.rule) == (MatchStatus.NEEDS_REVIEW, "exact_cross_document")


def test_known_alias_in_same_document_merges() -> None:
    r = EntityResolver([USJ_EXISTING], {"ENT-usj": {"DOC-2"}}).resolve(
        [mention("USJ", doc="DOC-2")]
    )
    assert r.mentions[0].entity_id == "ENT-usj"


def test_global_name_match_policy_is_the_explicit_baseline() -> None:
    """E5 baseline: name-only merging across documents, only when asked for."""
    r = EntityResolver([USJ_EXISTING], {"ENT-usj": {"DOC-1"}}, policy="global_name_match").resolve(
        [mention("USJ", doc="DOC-2")]
    )
    assert r.mentions[0].entity_id == "ENT-usj"
    assert [c.rule for c in r.candidates] == ["exact"]


def test_same_name_in_two_documents_gives_two_entities() -> None:
    a = mention("Project Alpha", EntityType.PROJECT, 0, doc="DOC-1")
    b = mention("Project Alpha", EntityType.PROJECT, 0, doc="DOC-2")
    r = EntityResolver().resolve([a, b])
    assert len(r.entities) == 2
    ids = {m.entity_id for m in r.mentions}
    assert len(ids) == 2
    assert {e.normalized_name for e in r.entities.values()} == {"project alpha"}
    review = [c for c in r.candidates if c.status is MatchStatus.NEEDS_REVIEW]
    assert [c.rule for c in review] == ["exact_cross_document"]


def test_alias_definition_pointing_to_other_document_is_review() -> None:
    r = EntityResolver([USJ_EXISTING.model_copy(update={"aliases": []})], {"ENT-usj": {"DOC-1"}})
    res = r.resolve(
        [mention("USJ", doc="DOC-2")],
        alias_definitions=[("University of Sri Jayewardenepura", "USJ")],
    )
    assert res.mentions[0].entity_id != "ENT-usj"
    assert any(
        c.rule == "alias_definition_cross_document" and c.status is MatchStatus.NEEDS_REVIEW
        for c in res.candidates
    )


def test_entity_id_is_stable_as_mentions_are_added() -> None:
    first = mention("Project Alpha", EntityType.PROJECT, 0)
    resolver = EntityResolver()
    r1 = resolver.resolve([first])
    (eid,) = r1.entities
    r2 = resolver.resolve([mention("Project Alpha", EntityType.PROJECT, 5)])
    assert r2.mentions[0].entity_id == eid
    assert eid == entity_id_for_mention(first.mention_id)


def test_duplicate_mentions_are_resolved_once() -> None:
    m = mention("Project Alpha", EntityType.PROJECT, 0)
    r = EntityResolver().resolve([m, m])
    assert len(r.mentions) == 1
    (e,) = r.entities.values()
    assert e.mention_ids == [m.mention_id]


def test_model_alias_only_creates_review_candidate() -> None:
    r = EntityResolver().resolve(
        [mention("Ministry of Education", n=0), mention("MoHE", n=1)],
        model_aliases=[("Ministry of Education", "MoHE")],
    )
    assert len(r.entities) == 2
    assert any(
        c.rule == "model_alias" and c.status is MatchStatus.NEEDS_REVIEW for c in r.candidates
    )


def test_resolution_is_deterministic() -> None:
    ms = [mention("USJ", n=1), mention("University of Sri Jayewardenepura", n=0)]
    a = EntityResolver().resolve(
        ms, alias_definitions=[("University of Sri Jayewardenepura", "USJ")]
    )
    b = EntityResolver().resolve(
        list(reversed(ms)), alias_definitions=[("University of Sri Jayewardenepura", "USJ")]
    )
    assert a.entities == b.entities
    assert a.mentions == b.mentions


# ------------------------------------------------- resolver final verification


def test_multiple_local_alias_definition_matches_never_merge() -> None:
    usj = Entity(
        entity_id="ENT-usj",
        canonical_name="University of Sri Jayewardenepura",
        normalized_name="university of sri jayewardenepura",
        entity_type=ORG,
    )
    usj2 = Entity(
        entity_id="ENT-usj2",
        canonical_name="Union of Software Journalists",
        normalized_name="union of software journalists",
        entity_type=ORG,
    )
    r = EntityResolver([usj, usj2], {"ENT-usj": {"DOC-1"}, "ENT-usj2": {"DOC-1"}}).resolve(
        [mention("USJ")],
        alias_definitions=[
            ("University of Sri Jayewardenepura", "USJ"),
            ("Union of Software Journalists", "USJ"),
        ],
    )
    (m,) = r.mentions
    assert m.entity_id is None
    assert {(c.entity_id, c.rule, c.status) for c in r.candidates} == {
        ("ENT-usj", "ambiguous_alias_definition", MatchStatus.NEEDS_REVIEW),
        ("ENT-usj2", "ambiguous_alias_definition", MatchStatus.NEEDS_REVIEW),
    }
    assert r.entities == {}  # nothing created or merged


def test_document_membership_is_updated_after_attach() -> None:
    """An entity founded in DOC-2 becomes local to DOC-2, so later DOC-2 mentions merge
    into it, not into the same-named entity from DOC-1."""
    r = EntityResolver([USJ_EXISTING], {"ENT-usj": {"DOC-1"}}).resolve(
        [
            mention("University of Sri Jayewardenepura", n=0, doc="DOC-2"),
            mention("USJ", n=1, doc="DOC-2"),
        ],
        alias_definitions=[("University of Sri Jayewardenepura", "USJ")],
    )
    long_m, short_m = sorted(r.mentions, key=lambda m: m.span_id)
    assert long_m.entity_id == short_m.entity_id != "ENT-usj"
    assert any(c.rule == "exact_cross_document" and c.entity_id == "ENT-usj" for c in r.candidates)


def test_default_policy_everywhere_is_document_scoped() -> None:
    import inspect

    from veritas.core.config import Settings
    from veritas.extraction.service import ExtractionService

    assert Settings().entity_resolution_policy == "document_scoped"
    assert inspect.signature(EntityResolver).parameters["policy"].default == "document_scoped"
    assert (
        inspect.signature(ExtractionService).parameters["resolution_policy"].default
        == "document_scoped"
    )

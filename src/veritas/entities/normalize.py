"""Deterministic name normalisation and alias-definition detection.

These functions decide nothing on their own. They produce the evidence that
``entities.resolver`` uses to choose between MERGED and NEEDS_REVIEW.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

STOPWORDS = frozenset({"of", "the", "and", "for", "in", "at", "on", "a", "an", "&"})
_PUNCT = re.compile(r"[^\w\s&]")
_WS = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """'The University of Sri Jayewardenepura.' -> 'university of sri jayewardenepura'.

    Case, punctuation, Unicode form and a leading 'the' are ignored. Word order and
    every other word are kept, so different names never normalise to the same key
    by accident.
    """
    s = unicodedata.normalize("NFKC", name).casefold()
    s = s.replace("&", " and ")
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s.removeprefix("the ")


def significant_tokens(name: str) -> frozenset[str]:
    return frozenset(t for t in normalize_name(name).split() if t not in STOPWORDS)


def acronym(name: str) -> str:
    """Initials of the significant words: 'University of Sri Jayewardenepura' -> 'USJ'."""
    words = [w for w in normalize_name(name).split() if w not in STOPWORDS]
    return "".join(w[0] for w in words).upper()


def looks_like_acronym(name: str) -> bool:
    s = name.strip().replace(".", "")
    return 2 <= len(s) <= 8 and s.isalnum() and s.upper() == s and any(c.isalpha() for c in s)


@dataclass(frozen=True)
class AliasDefinition:
    """'Long Name (SHORT)' found verbatim in a source text."""

    long_form: str
    short_form: str
    long_start: int
    short_start: int


# Capitalised phrase (allowing lower-case connectors) followed by "(ACRONYM)".
_DEFINITION = re.compile(
    r"((?:[A-Z][\w'’-]*)(?:\s+(?:of|the|and|for|&|[A-Z][\w'’-]*))*)\s*\(\s*([A-Z][A-Za-z0-9&.]{1,9})\s*\)"
)


def find_alias_definitions(text: str) -> list[AliasDefinition]:
    """Explicit 'Name (ACR)' definitions where ACR really is the name's acronym.

    The longest trailing word sequence whose initials equal ACR is taken as the long
    form, so 'Research Office of the University of Sri Jayewardenepura (USJ)' yields
    'University of Sri Jayewardenepura'. Parentheses that are not acronyms of the
    preceding words (e.g. '(LKR)', '(2025)') are ignored.
    """
    found: list[AliasDefinition] = []
    for m in _DEFINITION.finditer(text):
        phrase, short = m.group(1), m.group(2).replace(".", "")
        words = list(re.finditer(r"\S+", phrase))
        for i in range(len(words)):
            if words[i].group(0).lower() in STOPWORDS:
                continue  # a long form never starts with a connector
            candidate = phrase[words[i].start() :]
            if acronym(candidate) == short.upper():
                found.append(
                    AliasDefinition(
                        long_form=candidate,
                        short_form=m.group(2),
                        long_start=m.start(1) + words[i].start(),
                        short_start=m.start(2),
                    )
                )
                break
    return found

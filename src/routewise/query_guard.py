"""Lexical guard that vetoes embedding matches the embedder cannot tell apart.

Sentence embeddings are nearly blind to word order, numbers and negation, so
"Celsius to Fahrenheit" and "Fahrenheit to Celsius" score ~0.99. This guard is a
cheap, deterministic second stage: a cache hit is accepted only if

  1. both queries contain the same numbers,
  2. both queries contain the same negation words (not, no, never, ...), and
  3. the content words they share appear in the same relative order.

It deliberately does NOT try to catch word substitutions ("iPhone" vs "Android");
those score much lower on embeddings and are handled by the similarity threshold.
"""

from __future__ import annotations

import re

_APOSTROPHE_NT = re.compile(r"n['\u2019]t\b")
_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did",
        "i", "me", "my", "you", "your", "we", "it", "its", "of", "to", "in", "on",
        "for", "and", "or", "can", "could", "would", "should", "will", "please",
        "give", "tell", "there", "that", "this", "with", "at", "by", "from", "as",
    }
)  # fmt: skip

NEGATIONS = frozenset({"not", "no", "never", "without", "cannot", "none", "nothing"})


def _stem(token: str) -> str:
    """Strip a plural 's' so 'calories' and 'calorie' compare equal."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    """Lowercased, stopword-free, lightly stemmed tokens in original order."""
    text = _APOSTROPHE_NT.sub(" not", text.lower())
    tokens: list[str] = []
    for raw in _TOKEN.findall(text):
        if raw in STOPWORDS:
            continue
        if len(raw) == 1 and not raw.isdigit():  # leftovers like the "s" in "what's"
            continue
        tokens.append(_stem(raw))
    return tokens


def _dedupe(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def queries_compatible(a: str, b: str) -> bool:
    """True if two queries pass the numbers, negation and word-order checks."""
    tokens_a, tokens_b = content_tokens(a), content_tokens(b)
    set_a, set_b = set(tokens_a), set(tokens_b)

    if {t for t in set_a if t.isdigit()} != {t for t in set_b if t.isdigit()}:
        return False
    if (set_a & NEGATIONS) != (set_b & NEGATIONS):
        return False

    shared = set_a & set_b
    order_a = [t for t in _dedupe(tokens_a) if t in shared]
    order_b = [t for t in _dedupe(tokens_b) if t in shared]
    return order_a == order_b

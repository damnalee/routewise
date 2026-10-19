"""Heuristic query-complexity scoring for model routing.

Why heuristics first: they add ~0 latency, need no training data, and every score is
explainable (each signal's contribution is returned). The `ComplexityScorer` protocol lets a
learned classifier replace this later without touching the router.

The score is a sum of independent signals clamped to [0, 1]:

  length        longer prompts tend to need more reasoning
  code          code/tech vocabulary, error vocabulary, symbols, or a fenced code block
  reasoning     distinct reasoning terms (compare, trade-off, justify, reconcile, ...)
  math          math keywords and expressions
  multi_part    several questions, numbered lists, or sequencing words
  context       amount of retrieved context in the prompt (RAG synthesis is harder)
  simple_lookup negative signal for short factual/lookup questions
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

_FENCE = "`" * 3
_WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")

_REASONING_RE = re.compile(
    r"\b(?:why|explain|compar(?:e|es|ing|ison|isons)|trade-?offs?|analy[sz](?:e|es|is|ing)"
    r"|design|evaluat(?:e|ion)|step[- ]by[- ]step|pros and cons|difference between"
    r"|optimi[sz](?:e|ation|ing)|architecture|critique|justif(?:y|ication)|recommend\w*"
    r"|implications?|strateg(?:y|ies)|reconcil\w*|conflict\w*|contradict\w*|disagree\w*"
    r"|synthesi[sz]e|walk me through|in-depth|comprehensive|thorough|essay|argue"
    r"|migrat(?:e|ion)|rollback|risks?|mitigat\w*|zero[- ]downtime|versus|estimat\w*"
    r"|assumptions?|root cause)\b",
    re.IGNORECASE,
)

_CODE_LANG_RE = re.compile(
    r"(?<!\w)(?:python|javascript|typescript|java|rust|golang|c\+\+|c#|sql|postgres(?:ql)?"
    r"|mysql|regex|bash|docker|kubernetes|git|fastapi|pydantic|django|flask|react|numpy"
    r"|pandas|pytorch|rayon)(?!\w)",
    re.IGNORECASE,
)
_CODE_WORD_RE = re.compile(
    r"\b(?:function|algorithm|refactor\w*|debug\w*|endpoint|unit tests?|implement\w*|code"
    r"|compile\w*|api)\b",
    re.IGNORECASE,
)
_CODE_SYMBOL_RE = re.compile(r"\bO\([^)\n]*\)|\w\(\)|=>|->|==|\bdef\s+\w+|\bimport\s+\w+")
_CODE_ERROR_RE = re.compile(
    r"\b(?:errors?|exceptions?|traceback|stack trace|segfault|crash\w*|panic\w*|fails"
    r"|failing|failed)\b",
    re.IGNORECASE,
)
_CODE_CATEGORY_RES = (_CODE_LANG_RE, _CODE_WORD_RE, _CODE_SYMBOL_RE, _CODE_ERROR_RE)

_MATH_KW_RE = re.compile(
    r"\b(?:integral|integrate|derivative|differentiate|theorem|lemma|prove|proof|derive"
    r"|eigen\w+|matrix|matrices|determinant|probability|combinatorics|polynomial|gradient"
    r"|logits?)\b",
    re.IGNORECASE,
)
_MATH_EXPR_RE = re.compile(r"\d\s*[+*/^=]\s*\d|\b[a-z]\^\d|\d\s+-\s+\d")

_SEQUENCE_RE = re.compile(
    r"\b(?:first|second|third|then|next|finally|additionally|also|afterwards)\b", re.IGNORECASE
)
_NUMBERED_RE = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+\S")

_SIMPLE_START_RE = re.compile(
    r"^\W*(?:what|who|when|where|which|define|translate|convert|spell|list|name"
    r"|how (?:many|much|old|tall|far|long))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Weights:
    """Tunable signal weights. Defaults were chosen by hand; see eval/evaluate_router.py."""

    length_max: float = 0.25
    length_saturation_words: int = 100
    code_per_category: float = 0.12
    code_cap: float = 0.36
    code_fenced: float = 0.40
    reasoning_per_term: float = 0.15
    reasoning_cap: float = 0.45
    math_keyword: float = 0.30
    math_expression: float = 0.10
    multi_part: float = 0.15
    context_max: float = 0.15
    context_saturation_chars: int = 8_000
    simple_lookup: float = 0.25
    simple_max_words: int = 12


@dataclass(frozen=True, slots=True)
class ComplexityResult:
    score: float
    signals: dict[str, float]


class ComplexityScorer(Protocol):
    def score(self, query: str, context_chars: int = 0) -> ComplexityResult: ...


class HeuristicScorer:
    def __init__(self, weights: Weights | None = None) -> None:
        self._w = weights or Weights()

    def score(self, query: str, context_chars: int = 0) -> ComplexityResult:
        """Score a query in [0, 1]. `context_chars` is the size of retrieved context."""
        if context_chars < 0:
            raise ValueError("context_chars must be >= 0")
        w = self._w
        text = query.strip()
        n_words = len(_WORD_RE.findall(text))
        fenced = _FENCE in text
        signals: dict[str, float] = {}

        signals["length"] = min(n_words / w.length_saturation_words, 1.0) * w.length_max

        if fenced:
            signals["code"] = w.code_fenced
        else:
            categories = sum(1 for rx in _CODE_CATEGORY_RES if rx.search(text))
            signals["code"] = min(categories * w.code_per_category, w.code_cap)

        terms = {m.group(0).lower() for m in _REASONING_RE.finditer(text)}
        signals["reasoning"] = min(len(terms) * w.reasoning_per_term, w.reasoning_cap)

        math = 0.0
        if _MATH_KW_RE.search(text):
            math += w.math_keyword
        if _MATH_EXPR_RE.search(text):
            math += w.math_expression
        signals["math"] = math

        sequence = {m.group(0).lower() for m in _SEQUENCE_RE.finditer(text)}
        numbered = len(_NUMBERED_RE.findall(text))
        is_multi = text.count("?") >= 2 or numbered >= 2 or len(sequence) >= 2
        signals["multi_part"] = w.multi_part if is_multi else 0.0

        signals["context"] = min(context_chars / w.context_saturation_chars, 1.0) * w.context_max

        is_simple = (
            not fenced
            and 0 < n_words <= w.simple_max_words
            and _SIMPLE_START_RE.search(text) is not None
        )
        signals["simple_lookup"] = -w.simple_lookup if is_simple else 0.0

        total = min(max(sum(signals.values()), 0.0), 1.0)
        return ComplexityResult(
            score=round(total, 4),
            signals={name: round(value, 4) for name, value in signals.items()},
        )

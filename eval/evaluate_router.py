"""Evaluate routing quality and estimated cost savings on a labelled query set.

Metrics per threshold:
  hard recall     share of "large" queries correctly escalated, which protects quality
  over-routing    share of "small" queries wrongly sent to the large model, which wastes cost
  est. savings    cost reduction vs. always using the large model, at reference prices

The recommended threshold maximises savings subject to hard recall >= MIN_HARD_RECALL,
because sending a hard query to the small model risks a bad answer, which is worse than
overpaying for an easy one.

Run:  python eval/evaluate_router.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from routewise.complexity import HeuristicScorer
from routewise.config import get_settings

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "eval" / "routing_queries.jsonl"
REPORT_PATH = ROOT / "docs" / "routing_eval.md"

AVG_INPUT_TOKENS = 600  # typical RAG prompt including retrieved context
AVG_OUTPUT_TOKENS = 250
MIN_HARD_RECALL = 0.90

BoolArray = NDArray[np.bool_]
ScoreArray = NDArray[np.float64]


def load_items() -> list[dict[str, Any]]:
    with DATA_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def request_cost(price_in: float, price_out: float) -> float:
    return (AVG_INPUT_TOKENS * price_in + AVG_OUTPUT_TOKENS * price_out) / 1_000_000


def evaluate(
    scores: ScoreArray, labels: BoolArray, threshold: float, small_cost: float, large_cost: float
) -> dict[str, float]:
    pred_large = scores >= threshold
    tp = int(np.sum(pred_large & labels))
    fn = int(np.sum(~pred_large & labels))
    fp = int(np.sum(pred_large & ~labels))
    tn = int(np.sum(~pred_large & ~labels))
    n = len(scores)
    routed = int(pred_large.sum()) * large_cost + int((~pred_large).sum()) * small_cost
    return {
        "threshold": round(threshold, 2),
        "accuracy": (tp + tn) / n,
        "hard_recall": tp / (tp + fn) if (tp + fn) else 1.0,
        "over_routing": fp / (fp + tn) if (fp + tn) else 0.0,
        "large_share": float(pred_large.mean()),
        "savings": 1.0 - routed / (n * large_cost),
    }


def fmt_row(r: dict[str, float]) -> str:
    return (
        f"| {r['threshold']:.2f} | {r['accuracy']:.2f} | {r['hard_recall']:.2f} "
        f"| {r['over_routing']:.2f} | {r['large_share']:.2f} | {r['savings']:.0%} |"
    )


def main() -> None:
    settings = get_settings()
    items = load_items()
    scorer = HeuristicScorer()
    queries = [str(i["query"]) for i in items]
    scores = np.array(
        [scorer.score(str(i["query"]), int(i.get("context_chars", 0))).score for i in items]
    )
    labels = np.array([i["tier"] == "large" for i in items])

    small_cost = request_cost(settings.small_price_in, settings.small_price_out)
    large_cost = request_cost(settings.large_price_in, settings.large_price_out)

    print(f"queries: {len(items)} ({int(labels.sum())} large, {int((~labels).sum())} small)")
    print(f"cost per request: small ${small_cost:.6f}, large ${large_cost:.6f}")
    print(f"score: small mean={scores[~labels].mean():.3f} max={scores[~labels].max():.3f}")
    print(f"score: large mean={scores[labels].mean():.3f} min={scores[labels].min():.3f}")

    rows = [
        evaluate(scores, labels, float(t), small_cost, large_cost)
        for t in np.arange(0.10, 0.601, 0.05)
    ]
    header = [
        "| threshold | accuracy | hard recall | easy over-routing | share to large "
        "| est. savings vs always-large |",
        "|---|---|---|---|---|---|",
    ]
    table = header + [fmt_row(r) for r in rows]
    print("\n" + "\n".join(table))

    safe = [r for r in rows if r["hard_recall"] >= MIN_HARD_RECALL]
    if safe:
        best = max(safe, key=lambda r: (r["savings"], r["threshold"]))
        print(
            f"\nRecommended threshold: {best['threshold']:.2f} "
            f"(hard recall {best['hard_recall']:.2f}, savings {best['savings']:.0%})"
        )
    else:
        print(f"\nNo threshold reaches hard recall >= {MIN_HARD_RECALL:.2f}.")

    cfg = evaluate(scores, labels, settings.router_threshold, small_cost, large_cost)
    print(
        f"\nConfigured threshold {settings.router_threshold:.2f}: accuracy {cfg['accuracy']:.2f}, "
        f"hard recall {cfg['hard_recall']:.2f}, over-routing {cfg['over_routing']:.2f}, "
        f"savings {cfg['savings']:.0%}"
    )

    pred_large = scores >= settings.router_threshold
    print("\nHard queries sent to the SMALL model, a quality risk:")
    for i in np.where(labels & ~pred_large)[0]:
        print(f"  {scores[i]:.3f}  {queries[i]}")
    print("\nEasy queries sent to the LARGE model, wasted cost:")
    for i in np.where(~labels & pred_large)[0]:
        print(f"  {scores[i]:.3f}  {queries[i]}")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(
        "# Routing evaluation\n\n"
        f"{len(items)} hand-labelled queries. Savings assume {AVG_INPUT_TOKENS} input / "
        f"{AVG_OUTPUT_TOKENS} output tokens per request at reference prices, 0% cache hits.\n\n"
        + "\n".join(table)
        + "\n"
    )
    print(f"\nSaved {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

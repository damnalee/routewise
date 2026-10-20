"""Generate the README charts from the evaluation data and the local metrics DB.

Run from the project root:  python scripts/make_charts.py
Needs:  pip install -e ".[dev,charts]"
The embedding model is cached after the first run. Output goes to docs/images/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from routewise.complexity import HeuristicScorer
from routewise.config import Settings, get_settings
from routewise.embeddings import SentenceTransformerEmbedder
from routewise.query_guard import queries_compatible
from routewise.tracker import CostModel, CostTracker

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
PAIRS_PATH = ROOT / "eval" / "threshold_pairs.jsonl"
QUERIES_PATH = ROOT / "eval" / "routing_queries.jsonl"

AVG_INPUT_TOKENS = 600
AVG_OUTPUT_TOKENS = 250

GREEN, RED, BLUE, ORANGE, GRAY = "#2e7d32", "#c62828", "#1565c0", "#ef6c00", "#757575"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save(fig: Any, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def precision_recall(pred: Any, truth: Any) -> tuple[float, float]:
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def legend_below(ax: Any, columns: int) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=columns, frameon=False)


def cache_charts(settings: Settings) -> None:
    pairs = load_jsonl(PAIRS_PATH)
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model, device=settings.embedding_device
    )
    left = embedder.embed([str(p["a"]) for p in pairs])
    right = embedder.embed([str(p["b"]) for p in pairs])
    scores = np.sum(left * right, axis=1)
    same = np.array([bool(p["same"]) for p in pairs])
    guard_ok = np.array([queries_compatible(str(p["a"]), str(p["b"])) for p in pairs])
    threshold = settings.cache_similarity_threshold

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(scores[same], bins=bins, alpha=0.75, color=GREEN, label="Same question")
    ax.hist(scores[~same], bins=bins, alpha=0.75, color=RED, label="Different question")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Cache threshold {threshold:.2f}")
    vetoed = np.where(~same & ~guard_ok)[0]
    if len(vetoed):
        worst = vetoed[np.argmax(scores[vetoed])]
        ax.annotate(
            f"Word-order swap scores {scores[worst]:.3f}\nVetoed by the lexical guard",
            xy=(scores[worst], 0.3),
            xytext=(0.28, 0.8),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "black"},
            fontsize=9,
        )
    ax.set_xlabel("Cosine similarity between the two questions")
    ax.set_ylabel("Number of question pairs")
    ax.set_title("Embedding similarity cannot separate every pair")
    legend_below(ax, 3)
    save(fig, "cache_score_distribution.png")

    thresholds = np.arange(0.50, 0.981, 0.02)
    emb_precision: list[float] = []
    guard_precision: list[float] = []
    guard_recall: list[float] = []
    for t in thresholds:
        by_score = scores >= t
        emb_precision.append(precision_recall(by_score, same)[0])
        precision, recall = precision_recall(by_score & guard_ok, same)
        guard_precision.append(precision)
        guard_recall.append(recall)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(thresholds, emb_precision, color=GRAY, linestyle="--", marker="o", ms=3,
            label="Precision, embedding only")  # fmt: skip
    ax.plot(thresholds, guard_precision, color=BLUE, marker="o", ms=3,
            label="Precision, with guard")  # fmt: skip
    ax.plot(thresholds, guard_recall, color=ORANGE, marker="o", ms=3,
            label="Recall, with guard")  # fmt: skip
    ax.axvline(threshold, color="black", linestyle="--", alpha=0.6)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Score")
    ax.set_title("Cache precision and recall by threshold")
    legend_below(ax, 3)
    save(fig, "cache_threshold_sweep.png")


def routing_charts(settings: Settings) -> None:
    items = load_jsonl(QUERIES_PATH)
    scorer = HeuristicScorer()
    scores = np.array(
        [scorer.score(str(i["query"]), int(i.get("context_chars", 0))).score for i in items]
    )
    large = np.array([i["tier"] == "large" for i in items])
    threshold = settings.router_threshold
    small_cost = (
        AVG_INPUT_TOKENS * settings.small_price_in + AVG_OUTPUT_TOKENS * settings.small_price_out
    ) / 1_000_000
    large_cost = (
        AVG_INPUT_TOKENS * settings.large_price_in + AVG_OUTPUT_TOKENS * settings.large_price_out
    ) / 1_000_000

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(scores[~large], bins=bins, alpha=0.75, color=BLUE, label="Labelled easy")
    ax.hist(scores[large], bins=bins, alpha=0.75, color=ORANGE, label="Labelled hard")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Router threshold {threshold:.2f}")
    ax.set_xlabel("Complexity score")
    ax.set_ylabel("Number of queries")
    ax.set_title("Complexity scores of the labelled queries")
    legend_below(ax, 3)
    save(fig, "routing_score_distribution.png")

    thresholds = np.arange(0.10, 0.601, 0.05)
    hard_recall: list[float] = []
    accuracy: list[float] = []
    over_routing: list[float] = []
    savings: list[float] = []
    for t in thresholds:
        pred = scores >= t
        hard_recall.append(float(np.sum(pred & large) / np.sum(large)))
        accuracy.append(float(np.mean(pred == large)))
        over_routing.append(float(np.sum(pred & ~large) / np.sum(~large)))
        routed = int(pred.sum()) * large_cost + int((~pred).sum()) * small_cost
        savings.append(1.0 - routed / (len(scores) * large_cost))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(thresholds, hard_recall, color=ORANGE, marker="o", ms=3, label="Hard-query recall")
    ax.plot(thresholds, accuracy, color=BLUE, marker="o", ms=3, label="Accuracy")
    ax.plot(thresholds, over_routing, color=RED, marker="o", ms=3, label="Easy over-routing")
    ax.plot(
        thresholds, savings, color=GREEN, marker="o", ms=3, label="Est. savings vs always-large"
    )
    ax.axvline(threshold, color="black", linestyle="--", alpha=0.6)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Router threshold")
    ax.set_ylabel("Fraction")
    ax.set_title("Routing quality and savings by threshold")
    legend_below(ax, 2)
    save(fig, "routing_threshold_sweep.png")


def live_chart(settings: Settings) -> bool:
    db_path = ROOT / settings.db_path
    if not db_path.exists():
        return False
    tracker = CostTracker(db_path, CostModel.from_settings(settings))
    summary = tracker.summary()
    tracker.close()
    if summary["served"] == 0:
        return False

    latency = summary["latency_s"]
    outcomes = [o for o in ("cache_hit", "small", "large") if o in latency]
    names = {"cache_hit": "Cache hit", "small": "Small model", "large": "Large model"}
    colors = {"cache_hit": GREEN, "small": BLUE, "large": ORANGE}
    avg_ms = [latency[o]["avg"] * 1000 for o in outcomes]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = left.bar([names[o] for o in outcomes], avg_ms, color=[colors[o] for o in outcomes])
    left.set_yscale("log")
    left.set_ylabel("Average latency, ms, log scale")
    left.set_title("Latency by outcome")
    for bar, value in zip(bars, avg_ms, strict=True):
        left.annotate(
            f"{value:,.0f} ms",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
        )

    spend = [summary["baseline_usd"], summary["cost_usd"]]
    cost_bars = right.bar(["Always large model", "RouteWise"], spend, color=[GRAY, GREEN])
    right.set_ylabel("Spend, USD")
    right.set_title(f"Spend is {summary['savings_pct']:.0%} lower")
    for bar, value in zip(cost_bars, spend, strict=True):
        right.annotate(
            f"${value:.4f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
        )
    fig.suptitle(f"Local metrics DB: {summary['requests']} requests, synthetic traffic")
    save(fig, "live_results.png")
    return True


def main() -> None:
    settings = get_settings()
    plt.switch_backend("Agg")
    cache_charts(settings)
    routing_charts(settings)
    if not live_chart(settings):
        print("skipped live_results.png: no metrics DB with requests yet")


if __name__ == "__main__":
    main()

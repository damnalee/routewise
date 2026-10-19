"""Sweep similarity thresholds on labelled query pairs to choose a safe cache threshold.

Reports two configurations side by side:
  * embedding only      (similarity >= threshold)
  * embedding + guard   (similarity >= threshold AND lexical guard passes)

A cache "false hit" (serving the wrong answer) is worse than a miss, so the recommended
threshold maximises recall among those with precision >= MIN_PRECISION.

Run:  python eval/tune_threshold.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from routewise.config import get_settings
from routewise.embeddings import SentenceTransformerEmbedder
from routewise.query_guard import queries_compatible

ROOT = Path(__file__).resolve().parent.parent
PAIRS_PATH = ROOT / "eval" / "threshold_pairs.jsonl"
REPORT_PATH = ROOT / "docs" / "threshold_sweep.md"
MIN_PRECISION = 0.95

BoolArray = NDArray[np.bool_]


def load_pairs() -> list[dict[str, object]]:
    with PAIRS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def metrics(pred: BoolArray, labels: BoolArray) -> tuple[float, float, float]:
    """Return (precision, recall, false-hit rate)."""
    tp = int(np.sum(pred & labels))
    fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels))
    n_neg = max(int(np.sum(~labels)), 1)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, fp / n_neg


def main() -> None:
    pairs = load_pairs()
    embedder = SentenceTransformerEmbedder(get_settings().embedding_model)
    a = embedder.embed([str(p["a"]) for p in pairs])
    b = embedder.embed([str(p["b"]) for p in pairs])
    scores = np.sum(a * b, axis=1)
    labels = np.array([bool(p["same"]) for p in pairs])
    guard_ok = np.array([queries_compatible(str(p["a"]), str(p["b"])) for p in pairs])

    print(f"model: {get_settings().embedding_model}")
    print(f"pairs: {len(pairs)} ({int(labels.sum())} paraphrase, {int((~labels).sum())} different)")
    print(f"paraphrase score: mean={scores[labels].mean():.3f} min={scores[labels].min():.3f}")
    print(f"different  score: mean={scores[~labels].mean():.3f} max={scores[~labels].max():.3f}")

    vetoed_neg = int(np.sum(~guard_ok & ~labels))
    vetoed_pos = int(np.sum(~guard_ok & labels))
    print(f"\nGuard vetoed {vetoed_neg} different-question pairs (good)")
    print(f"Guard vetoed {vetoed_pos} paraphrase pairs (recall cost)")
    for i in np.where(~guard_ok)[0]:
        tag = "PARAPHRASE" if labels[i] else "different "
        print(f"  [{tag}] {scores[i]:.3f}  {pairs[i]['a']!r}  vs  {pairs[i]['b']!r}")

    rows: list[tuple[float, float, float, float, float, float]] = []
    for t in np.arange(0.50, 0.981, 0.02):
        by_score = scores >= t
        p_e, r_e, _ = metrics(by_score, labels)
        p_g, r_g, fh_g = metrics(by_score & guard_ok, labels)
        rows.append((round(float(t), 2), p_e, r_e, p_g, r_g, fh_g))

    lines = [
        "| threshold | precision (emb) | recall (emb) | precision (+guard) | recall (+guard) "
        "| false-hit rate (+guard) |",
        "|---|---|---|---|---|---|",
        *(f"| {t:.2f} | {pe:.2f} | {re:.2f} | {pg:.2f} | {rg:.2f} | {fh:.2f} |"
          for t, pe, re, pg, rg, fh in rows),
    ]  # fmt: skip
    print("\n" + "\n".join(lines))

    safe = [r for r in rows if r[3] >= MIN_PRECISION and r[4] > 0]
    best = max(safe, key=lambda r: (r[4], r[0])) if safe else None
    if best:
        print(
            f"\nRecommended threshold (with guard): {best[0]:.2f} "
            f"(precision {best[3]:.2f}, recall {best[4]:.2f})"
        )
    else:
        print(f"\nNo threshold reaches precision >= {MIN_PRECISION} even with the guard.")

    print("\nHardest negatives that still pass the guard (the threshold must stop these):")
    survivors = np.where(~labels & guard_ok, scores, -1.0)
    for i in np.argsort(-survivors)[:5]:
        if survivors[i] < 0:
            break
        print(f"  {scores[i]:.3f}  {pairs[i]['a']!r}  vs  {pairs[i]['b']!r}")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("# Cache threshold sweep\n\n" + "\n".join(lines) + "\n")
    print(f"\nSaved {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

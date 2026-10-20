"""Send mixed synthetic traffic to a running RouteWise gateway so the dashboard has data.

Mix: new easy questions, exact repeats, light rewordings and paraphrases of earlier ones, and
hard reasoning questions. Hit rate and savings shown in the dashboard come from the gateway;
the tally printed here is only a client-side sanity check.

Run:  python scripts/simulate_traffic.py --n 60
Note: hard questions on a reasoning model can use ~2,500 output tokens each.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import httpx

from routewise.config import get_settings

EASY_PAIRS = [
    ("How can I reset my password?", "What's the way to reset my password?"),
    ("What is the capital of France?", "Which city is the capital of France?"),
    ("Explain how a stack works", "Can you explain how stacks work?"),
    ("Who wrote Hamlet?", "Who is the author of Hamlet?"),
    ("How do I delete a git branch?", "What command removes a git branch?"),
    ("What is semantic caching?", "Can you explain what semantic caching is?"),
    ("How many calories are in an apple?", "What is the calorie count of one apple?"),
    ("What is the time complexity of quicksort?", "How fast is quicksort in big-O terms?"),
    ("How do I install Python on Mac?", "What are the steps to install Python on macOS?"),
    ("What are the symptoms of the flu?", "Which symptoms does influenza cause?"),
]
HARD = [
    "Design a scalable architecture for a multi-tenant SaaS platform and explain the trade-offs "
    "between a shared database and database-per-tenant.",
    "Compare the pros and cons of microservices versus a monolith for a team of five engineers, "
    "and recommend one with justification.",
    "Plan a migration from MySQL to PostgreSQL with zero downtime, listing the risks and rollback "
    "steps.",
    "Explain why the CAP theorem implies we cannot have both consistency and availability during "
    "a partition, and how Cassandra and Spanner make different trade-offs.",
    "What are the security implications of storing JWTs in localStorage versus HttpOnly cookies, "
    "and how would you mitigate them?",
    "Design a rate limiter that supports burst traffic and per-user quotas, explain the algorithm "
    "choice, and write the code.",
]

POOL = [q for pair in EASY_PAIRS for q in pair]
PARAPHRASE = {a: b for a, b in EASY_PAIRS} | {b: a for a, b in EASY_PAIRS}


def light_variant(query: str, rng: random.Random) -> str:
    kind = rng.choice(["lower", "please", "no_punct"])
    if kind == "lower":
        return query.lower()
    if kind == "please":
        return query.rstrip("?.!") + ", please?"
    return query.rstrip("?.!")


def build_plan(
    n: int, hard_ratio: float, repeat_ratio: float, concurrency: int, rng: random.Random
) -> list[str]:
    plan: list[str] = []
    used: list[str] = []
    for _ in range(n):
        if rng.random() < hard_ratio:
            plan.append(rng.choice(HARD))
            continue
        # Skip the most recent items so a repeat never races its own original request.
        candidates = used[:-concurrency] if len(used) > concurrency else []
        if candidates and rng.random() < repeat_ratio:
            base = rng.choice(candidates)
            roll = rng.random()
            if roll < 0.4:
                plan.append(base)
            elif roll < 0.7:
                plan.append(light_variant(base, rng))
            else:
                plan.append(PARAPHRASE[base])
            continue
        unused = [q for q in POOL if q not in used and PARAPHRASE[q] not in used]
        base = rng.choice(unused or POOL)
        used.append(base)
        plan.append(base)
    return plan


@dataclass(slots=True)
class Result:
    source: str
    cost: float = 0.0
    saved: float = 0.0
    latency_ms: float = 0.0


async def send(client: httpx.AsyncClient, query: str, max_tokens: int | None) -> Result:
    body: dict[str, Any] = {"query": query}
    if max_tokens:
        body["max_tokens"] = max_tokens
    start = time.perf_counter()
    try:
        response = await client.post("/v1/query", json=body)
    except httpx.HTTPError as exc:
        return Result(source=f"error:{type(exc).__name__}")
    elapsed_ms = (time.perf_counter() - start) * 1000
    if response.status_code != 200:
        try:
            code = response.json()["error"]["code"]
        except (ValueError, KeyError, TypeError):
            code = str(response.status_code)
        return Result(source=f"error:{code}", latency_ms=elapsed_ms)
    data = response.json()
    source = data["source"] + (" (degraded)" if data.get("degraded") else "")
    return Result(
        source=source,
        cost=data["cost_usd"],
        saved=data["saved_usd"],
        latency_ms=data["latency_ms"],
    )


async def run(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    plan = build_plan(args.n, args.hard_ratio, args.repeat_ratio, args.concurrency, rng)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=120) as client:
        try:
            ready = await client.get("/readyz")
        except httpx.HTTPError:
            print(f"Gateway not reachable at {args.base_url}. Start it first.")
            return 1
        if ready.status_code != 200:
            print("Warning: /readyz is not OK, the LLM provider may be down.")

        semaphore = asyncio.Semaphore(args.concurrency)
        done = 0

        async def one(query: str) -> Result:
            nonlocal done
            async with semaphore:
                result = await send(client, query, args.max_tokens)
                if args.delay:
                    await asyncio.sleep(args.delay)
            done += 1
            if done % 10 == 0 or done == len(plan):
                print(f"  {done}/{len(plan)} done")
            return result

        print(f"Sending {len(plan)} requests to {args.base_url} (concurrency {args.concurrency})")
        results = await asyncio.gather(*(one(q) for q in plan))

    counts = Counter(r.source for r in results)
    latency: dict[str, list[float]] = defaultdict(list)
    for r in results:
        latency[r.source].append(r.latency_ms)
    print("\nResult by source:")
    for source, count in counts.most_common():
        avg = sum(latency[source]) / len(latency[source])
        print(f"  {source:28s} {count:4d}   avg {avg:8.0f} ms")
    print(
        f"\nSpend ${sum(r.cost for r in results):.4f}   saved ${sum(r.saved for r in results):.4f}"
    )
    print("Open the dashboard: python -m streamlit run src/routewise/dashboard.py")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=60, help="number of requests")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--hard-ratio", type=float, default=0.25)
    parser.add_argument("--repeat-ratio", type=float, default=0.5)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=get_settings().gateway_api_key)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to pause per request")
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))

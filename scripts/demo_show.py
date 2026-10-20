"""Pretty-print one gateway response for scripts/demo.sh.

Reads curl output (JSON body, then a line "HTTP <code>") from stdin. The optional first
argument lists acceptable outcomes separated by commas, for example "cache,small" or
"http422". Exits 1 when the outcome is not one of them.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def describe(data: dict[str, Any], code: str) -> tuple[str, str]:
    """Return (outcome, human readable text)."""
    if "error" in data:
        err = data["error"]
        return f"http{code}", f"HTTP {code} {err.get('code')}: {err.get('message')}"
    if "detail" in data:
        return f"http{code}", f"HTTP {code} request validation failed"

    source = str(data.get("source"))
    parts = [f"source={source}", f"model={data.get('model')}"]
    if data.get("cache_score") is not None:
        parts.append(f"cache_score={data['cache_score']}")
    if data.get("complexity_score") is not None:
        parts.append(f"complexity={data['complexity_score']}")
    if data.get("degraded"):
        parts.append("DEGRADED")
    parts.append(f"cost=${data.get('cost_usd', 0):.6f}")
    parts.append(f"saved=${data.get('saved_usd', 0):.6f}")
    parts.append(f"{data.get('latency_ms', 0):.0f}ms")
    answer = " ".join(str(data.get("answer", "")).split())[:80]
    return source, "  ".join(parts) + f'\n      answer: "{answer}"'


def main() -> int:
    expected = {item for item in (sys.argv[1].split(",") if len(sys.argv) > 1 else []) if item}
    raw = sys.stdin.read()
    payload, _, code = raw.rpartition("\nHTTP ")
    code = code.strip()
    try:
        data = json.loads(payload)
    except ValueError:
        print(f"   [UNEXPECTED] HTTP {code}: {payload[:200]!r}")
        return 1
    outcome, text = describe(data, code)
    ok = not expected or outcome in expected
    print(f"   {'[ok]' if ok else '[UNEXPECTED]'} {text}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

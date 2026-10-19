"""Groq client (OpenAI-compatible chat completions) with multi-key failover.

Key handling:
  * Keys are used round-robin to spread load.
  * HTTP 429 puts that key on cooldown (Retry-After) and the next key is tried at once.
  * HTTP 401/403 disables that key for the process lifetime.
  * If every key is cooling down, wait for the soonest one (capped), else raise.

Other errors follow the same rules as OllamaClient: 5xx, timeouts and transport errors are
retried with exponential backoff, 404 and other 4xx are not.

Note: gpt-oss models are reasoning models. Reasoning tokens count against max_tokens, so a
tiny max_tokens can yield an empty answer; that is reported as LLMBadResponseError.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx

from routewise.config import Settings
from routewise.llm import (
    ChatMessage,
    LLMBadResponseError,
    LLMClient,
    LLMError,
    LLMModelNotFoundError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    _validate_request,
    estimate_tokens,
)
from routewise.logging import get_logger

log = get_logger(__name__)

_DEFAULT_COOLDOWN_S = 5.0
_MAX_COOLDOWN_S = 60.0
_MAX_WAIT_S = 30.0


class LLMRateLimitError(LLMError):
    pass


class LLMAuthError(LLMError):
    pass


class KeyPool:
    """Round-robin API key pool with per-key cooldown and permanent disable."""

    def __init__(self, keys: Sequence[str], clock: Callable[[], float] = time.monotonic) -> None:
        cleaned = [k for k in keys if k]
        if not cleaned:
            raise ValueError("at least one API key is required (set ROUTEWISE_GROQ_API_KEYS)")
        self._keys = cleaned
        self._clock = clock
        self._blocked_until = [0.0] * len(cleaned)
        self._disabled = [False] * len(cleaned)
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def all_disabled(self) -> bool:
        return all(self._disabled)

    def acquire(self) -> tuple[int, str] | None:
        """Next usable key as (index, key), or None if all are cooling down or disabled."""
        now = self._clock()
        n = len(self._keys)
        for offset in range(n):
            idx = (self._cursor + offset) % n
            if not self._disabled[idx] and self._blocked_until[idx] <= now:
                self._cursor = (idx + 1) % n
                return idx, self._keys[idx]
        return None

    def cooldown(self, idx: int, seconds: float) -> None:
        self._blocked_until[idx] = self._clock() + seconds

    def disable(self, idx: int) -> None:
        self._disabled[idx] = True

    def seconds_until_available(self) -> float:
        now = self._clock()
        waits = [
            max(self._blocked_until[i] - now, 0.0)
            for i in range(len(self._keys))
            if not self._disabled[i]
        ]
        return min(waits) if waits else 0.0

    def first_key(self) -> str:
        return self._keys[0]


def _retry_after_s(response: httpx.Response) -> float:
    raw = response.headers.get("retry-after")
    try:
        value = float(raw) if raw is not None else _DEFAULT_COOLDOWN_S
    except ValueError:
        value = _DEFAULT_COOLDOWN_S
    return min(max(value, 0.0), _MAX_COOLDOWN_S)


class GroqClient(LLMClient):
    def __init__(
        self,
        api_keys: Sequence[str],
        base_url: str,
        timeout_s: float,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        reasoning_effort: str | None = "low",
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_retries < 0 or backoff_s < 0:
            raise ValueError("max_retries and backoff_s must be >= 0")
        self._pool = KeyPool(api_keys, clock=clock)
        self._base_url = base_url
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._reasoning_effort = reasoning_effort
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> GroqClient:
        return cls(
            api_keys=settings.groq_keys,
            base_url=settings.groq_base_url,
            timeout_s=settings.request_timeout_s,
            max_retries=settings.llm_max_retries,
            backoff_s=settings.llm_retry_backoff_s,
            reasoning_effort=settings.groq_reasoning_effort,
        )

    async def generate(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        _validate_request(messages, temperature)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens
        if self._reasoning_effort and model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = self._reasoning_effort

        start = time.perf_counter()
        last_error: LLMError | None = None
        for attempt in range(self._max_retries + 1):
            picked = self._pool.acquire()
            if picked is None:
                if self._pool.all_disabled:
                    raise LLMAuthError("all Groq API keys were rejected (HTTP 401/403)")
                last_error = LLMRateLimitError("all Groq API keys are rate limited")
                if attempt < self._max_retries:
                    wait = min(self._pool.seconds_until_available(), _MAX_WAIT_S)
                    log.warning("llm.all_keys_cooling", wait_s=round(wait, 2))
                    await self._sleep(wait)
                    continue
                raise last_error

            idx, key = picked
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {key}"},
                )
            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(f"timeout calling {model}: {exc!r}")
            except httpx.TransportError as exc:
                last_error = LLMUnavailableError(f"cannot reach Groq at {self._base_url}: {exc!r}")
            else:
                status = response.status_code
                if status in (401, 403):
                    self._pool.disable(idx)
                    last_error = LLMAuthError(f"Groq key #{idx + 1} rejected (HTTP {status})")
                    log.error("llm.key_disabled", key_index=idx + 1, status=status)
                    continue
                if status == 429:
                    cooldown = _retry_after_s(response)
                    self._pool.cooldown(idx, cooldown)
                    last_error = LLMRateLimitError(f"Groq key #{idx + 1} rate limited")
                    log.warning("llm.rate_limited", key_index=idx + 1, cooldown_s=cooldown)
                    continue
                if status == 404:
                    raise LLMModelNotFoundError(
                        f"model {model!r} not found on Groq, check ROUTEWISE_SMALL_MODEL "
                        "and ROUTEWISE_LARGE_MODEL against GET /models"
                    )
                if status >= 500:
                    last_error = LLMUnavailableError(f"Groq returned HTTP {status}")
                elif status >= 400:
                    raise LLMBadResponseError(
                        f"Groq rejected request: HTTP {status}: {response.text[:300]}"
                    )
                else:
                    result = self._parse(response, model, messages, time.perf_counter() - start)
                    log.debug(
                        "llm.ok",
                        model=model,
                        tokens_in=result.tokens_in,
                        tokens_out=result.tokens_out,
                        latency_s=round(result.latency_s, 3),
                        attempts=attempt + 1,
                        key_index=idx + 1,
                    )
                    return result

            if attempt < self._max_retries:
                delay = self._backoff_s * 2**attempt
                log.warning(
                    "llm.retry",
                    model=model,
                    attempt=attempt + 1,
                    delay_s=delay,
                    error=str(last_error),
                )
                await self._sleep(delay)

        assert last_error is not None
        raise last_error

    @staticmethod
    def _parse(
        response: httpx.Response,
        model: str,
        messages: Sequence[ChatMessage],
        latency_s: float,
    ) -> LLMResponse:
        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMBadResponseError(f"malformed Groq response: {exc!r}") from exc
        if not isinstance(text, str) or not text.strip():
            reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            raise LLMBadResponseError(
                f"empty completion (finish_reason={reason}); a reasoning model may need a "
                "larger max_tokens"
            )

        usage = data.get("usage") if isinstance(data, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        reported_in = usage.get("prompt_tokens")
        reported_out = usage.get("completion_tokens")
        tokens_in = (
            reported_in
            if isinstance(reported_in, int)
            else estimate_tokens(" ".join(m.content for m in messages))
        )
        tokens_out = reported_out if isinstance(reported_out, int) else estimate_tokens(text)
        estimated = not (isinstance(reported_in, int) and isinstance(reported_out, int))
        return LLMResponse(
            text=text,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=latency_s,
            estimated_tokens=estimated,
        )

    async def healthy(self) -> bool:
        try:
            response = await self._client.get(
                "/models", headers={"Authorization": f"Bearer {self._pool.first_key()}"}
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def available_models(self) -> set[str]:
        """Model IDs the first key can use. Handy for startup checks."""
        try:
            response = await self._client.get(
                "/models", headers={"Authorization": f"Bearer {self._pool.first_key()}"}
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailableError(f"cannot list Groq models: {exc!r}") from exc
        items = data.get("data", []) if isinstance(data, dict) else []
        return {str(m["id"]) for m in items if isinstance(m, dict) and "id" in m}

    async def aclose(self) -> None:
        await self._client.aclose()

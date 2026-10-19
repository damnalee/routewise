"""LLM provider clients.

`LLMClient` is the interface the gateway depends on. `OllamaClient` talks to a local Ollama
server over HTTP. `FakeLLMClient` is a deterministic stand-in for tests and offline dev.

Error model (all subclasses of `LLMError`):
  LLMUnavailableError     server unreachable or HTTP 5xx, retried with exponential backoff
  LLMTimeoutError         request timed out, retried with exponential backoff
  LLMModelNotFoundError   HTTP 404, model not pulled, never retried
  LLMBadResponseError     HTTP 4xx or malformed body, never retried

Latency is wall-clock time for the whole call, including retries and backoff, because that
is what the end user experiences.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from routewise.config import Settings
from routewise.logging import get_logger

log = get_logger(__name__)


class LLMError(Exception):
    """Base class for provider failures."""


class LLMUnavailableError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMModelNotFoundError(LLMError):
    pass


class LLMBadResponseError(LLMError):
    pass


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    estimated_tokens: bool = False  # True when the provider did not report token counts


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token) used only as a fallback."""
    return max(1, len(text) // 4)


def _validate_request(messages: Sequence[ChatMessage], temperature: float) -> None:
    if not messages:
        raise ValueError("messages must not be empty")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be in [0, 2]")


class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def healthy(self) -> bool: ...

    async def aclose(self) -> None:
        return None


class OllamaClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        timeout_s: float,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_retries < 0 or backoff_s < 0:
            raise ValueError("max_retries and backoff_s must be >= 0")
        self._base_url = base_url
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaClient:
        return cls(
            base_url=settings.llm_base_url,
            timeout_s=settings.request_timeout_s,
            max_retries=settings.llm_max_retries,
            backoff_s=settings.llm_retry_backoff_s,
        )

    async def generate(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        _validate_request(messages, temperature)
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }

        start = time.perf_counter()
        last_error: LLMError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post("/api/chat", json=payload)
            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(f"timeout calling {model}: {exc!r}")
            except httpx.TransportError as exc:
                last_error = LLMUnavailableError(
                    f"cannot reach Ollama at {self._base_url}: {exc!r}"
                )
            else:
                status = response.status_code
                if status == 404:
                    raise LLMModelNotFoundError(
                        f"model {model!r} not found, run: ollama pull {model}"
                    )
                if status >= 500:
                    last_error = LLMUnavailableError(f"Ollama returned HTTP {status}")
                elif status >= 400:
                    raise LLMBadResponseError(
                        f"Ollama rejected request: HTTP {status}: {response.text[:200]}"
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
        data: Any
        try:
            data = response.json()
            text = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMBadResponseError(f"malformed Ollama response: {exc!r}") from exc
        if not isinstance(text, str):
            raise LLMBadResponseError("Ollama response content is not a string")

        reported_in = data.get("prompt_eval_count")
        reported_out = data.get("eval_count")
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
            response = await self._client.get("/api/tags")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def available_models(self) -> set[str]:
        """Names of models pulled on the server. Used for startup checks."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailableError(f"cannot list models: {exc!r}") from exc
        models = data.get("models", []) if isinstance(data, dict) else []
        return {str(m["name"]) for m in models if isinstance(m, dict) and "name" in m}

    async def aclose(self) -> None:
        await self._client.aclose()


class FakeLLMClient(LLMClient):
    """Deterministic client for tests and offline development. Records every call."""

    def __init__(
        self,
        reply: str = "fake answer",
        tokens_out: int = 20,
        latency_s: float = 0.0,
        fail_with: LLMError | None = None,
    ) -> None:
        self._reply = reply
        self._tokens_out = tokens_out
        self._latency_s = latency_s
        self._fail_with = fail_with
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def generate(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        _validate_request(messages, temperature)
        self.calls.append((model, list(messages)))
        if self._fail_with is not None:
            raise self._fail_with
        return LLMResponse(
            text=self._reply,
            model=model,
            tokens_in=estimate_tokens(" ".join(m.content for m in messages)),
            tokens_out=self._tokens_out,
            latency_s=self._latency_s,
        )

    async def healthy(self) -> bool:
        return self._fail_with is None

import json
from typing import Any

import httpx
import pytest

from routewise.config import Settings
from routewise.llm import (
    ChatMessage,
    LLMBadResponseError,
    LLMModelNotFoundError,
    OllamaClient,
)
from routewise.llm_factory import build_llm_client
from routewise.llm_groq import GroqClient, KeyPool, LLMAuthError, LLMRateLimitError

MESSAGES = [ChatMessage("user", "Say hi")]
SMALL = "openai/gpt-oss-20b"


def ok_body(text: str = "hi", tokens_in: int = 10, tokens_out: int = 4) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }


def make_client(
    handler: Any, keys: tuple[str, ...] = ("k1", "k2"), max_retries: int = 2
) -> tuple[GroqClient, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = GroqClient(
        api_keys=keys,
        base_url="https://groq.test/openai/v1",
        timeout_s=5,
        max_retries=max_retries,
        backoff_s=0.5,
        reasoning_effort="low",
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
    )
    return client, sleeps


async def test_success_sends_expected_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_body("hello", 12, 5))

    client, sleeps = make_client(handler)
    result = await client.generate(SMALL, MESSAGES, temperature=0.1, max_tokens=64)

    assert (result.text, result.tokens_in, result.tokens_out) == ("hello", 12, 5)
    assert result.estimated_tokens is False and sleeps == []
    request = seen[0]
    body = json.loads(request.content)
    assert request.url.path == "/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k1"
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == "low"
    assert body["stream"] is False
    await client.aclose()


async def test_reasoning_effort_only_for_gpt_oss_models() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    client, _ = make_client(handler)
    await client.generate("some/other-model", MESSAGES)
    assert "reasoning_effort" not in seen[0]
    await client.aclose()


async def test_keys_rotate_round_robin() -> None:
    used: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        used.append(request.headers["authorization"])
        return httpx.Response(200, json=ok_body())

    client, _ = make_client(handler)
    await client.generate(SMALL, MESSAGES)
    await client.generate(SMALL, MESSAGES)
    await client.generate(SMALL, MESSAGES)
    assert used == ["Bearer k1", "Bearer k2", "Bearer k1"]
    await client.aclose()


async def test_429_fails_over_to_next_key_without_sleeping() -> None:
    used: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        used.append(request.headers["authorization"])
        if request.headers["authorization"] == "Bearer k1":
            return httpx.Response(429, headers={"retry-after": "2"})
        return httpx.Response(200, json=ok_body())

    client, sleeps = make_client(handler)
    result = await client.generate(SMALL, MESSAGES)
    assert result.text == "hi"
    assert used == ["Bearer k1", "Bearer k2"] and sleeps == []
    await client.aclose()


async def test_all_keys_rate_limited_raises() -> None:
    client, _ = make_client(lambda request: httpx.Response(429, headers={"retry-after": "30"}))
    with pytest.raises(LLMRateLimitError):
        await client.generate(SMALL, MESSAGES)
    await client.aclose()


async def test_401_disables_key_and_falls_back() -> None:
    used: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        used.append(request.headers["authorization"])
        if request.headers["authorization"] == "Bearer k1":
            return httpx.Response(401, json={"error": "invalid key"})
        return httpx.Response(200, json=ok_body())

    client, _ = make_client(handler)
    await client.generate(SMALL, MESSAGES)
    await client.generate(SMALL, MESSAGES)
    assert used == ["Bearer k1", "Bearer k2", "Bearer k2"]
    await client.aclose()


async def test_all_keys_rejected_raises_auth_error() -> None:
    client, _ = make_client(lambda request: httpx.Response(401))
    with pytest.raises(LLMAuthError):
        await client.generate(SMALL, MESSAGES)
    await client.aclose()


async def test_5xx_retries_with_backoff_then_succeeds() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, json=ok_body())

    client, sleeps = make_client(handler, keys=("k1",), max_retries=2)
    result = await client.generate(SMALL, MESSAGES)
    assert result.text == "hi" and len(calls) == 3
    assert sleeps == [0.5, 1.0]
    await client.aclose()


async def test_model_not_found_is_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"error": {"code": "model_not_found"}})

    client, _ = make_client(handler)
    with pytest.raises(LLMModelNotFoundError):
        await client.generate("ghost/model", MESSAGES)
    assert len(calls) == 1
    await client.aclose()


async def test_empty_completion_raises_bad_response() -> None:
    body = ok_body(text="")
    body["choices"][0]["finish_reason"] = "length"
    client, _ = make_client(lambda request: httpx.Response(200, json=body))
    with pytest.raises(LLMBadResponseError, match="max_tokens"):
        await client.generate(SMALL, MESSAGES, max_tokens=5)
    await client.aclose()


async def test_missing_usage_falls_back_to_estimates() -> None:
    body = {"choices": [{"message": {"content": "x" * 40}, "finish_reason": "stop"}]}
    client, _ = make_client(lambda request: httpx.Response(200, json=body))
    result = await client.generate(SMALL, MESSAGES)
    assert result.estimated_tokens is True and result.tokens_out == 10
    await client.aclose()


async def test_healthy_and_available_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/v1/models"
        return httpx.Response(200, json={"data": [{"id": SMALL}, {"id": "openai/gpt-oss-120b"}]})

    client, _ = make_client(handler)
    assert await client.healthy() is True
    assert await client.available_models() == {SMALL, "openai/gpt-oss-120b"}
    await client.aclose()


def test_no_keys_is_rejected() -> None:
    with pytest.raises(ValueError, match="ROUTEWISE_GROQ_API_KEYS"):
        GroqClient(api_keys=[], base_url="https://groq.test", timeout_s=5)


def test_key_pool_cooldown_and_recovery() -> None:
    now = [0.0]
    pool = KeyPool(["a", "b"], clock=lambda: now[0])
    assert pool.acquire() == (0, "a")
    assert pool.acquire() == (1, "b")
    pool.cooldown(0, 10)
    pool.cooldown(1, 5)
    assert pool.acquire() is None
    assert pool.seconds_until_available() == pytest.approx(5)
    now[0] = 6.0
    assert pool.acquire() == (1, "b")
    pool.disable(1)
    now[0] = 11.0
    assert pool.acquire() == (0, "a")
    pool.disable(0)
    assert pool.all_disabled is True


def test_settings_parse_keys_and_factory_picks_provider() -> None:
    settings = Settings(_env_file=None, groq_api_keys=" a, b ,,c")
    assert settings.groq_keys == ["a", "b", "c"]
    assert isinstance(build_llm_client(settings), GroqClient)
    assert isinstance(
        build_llm_client(Settings(_env_file=None, llm_provider="ollama")), OllamaClient
    )

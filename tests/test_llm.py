import json
from typing import Any

import httpx
import pytest

from routewise.llm import (
    ChatMessage,
    FakeLLMClient,
    LLMBadResponseError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    LLMUnavailableError,
    OllamaClient,
)

MESSAGES = [ChatMessage("user", "Say hi")]


def ok_body(text: str = "hi", tokens_in: int = 12, tokens_out: int = 5) -> dict[str, Any]:
    return {
        "model": "m",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "prompt_eval_count": tokens_in,
        "eval_count": tokens_out,
    }


def make_client(handler: Any, max_retries: int = 2) -> tuple[OllamaClient, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = OllamaClient(
        base_url="http://ollama.test",
        timeout_s=5,
        max_retries=max_retries,
        backoff_s=0.5,
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
    )
    return client, sleeps


async def test_success_returns_text_tokens_and_sends_expected_payload() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body("hello there", 12, 5))

    client, sleeps = make_client(handler)
    result = await client.generate("llama3.2:3b", MESSAGES, temperature=0.1, max_tokens=64)

    assert result.text == "hello there"
    assert (result.tokens_in, result.tokens_out) == (12, 5)
    assert result.estimated_tokens is False
    assert result.model == "llama3.2:3b"
    assert result.latency_s >= 0
    assert sleeps == []
    assert seen[0]["stream"] is False
    assert seen[0]["model"] == "llama3.2:3b"
    assert seen[0]["options"] == {"temperature": 0.1, "num_predict": 64}
    assert seen[0]["messages"] == [{"role": "user", "content": "Say hi"}]
    await client.aclose()


async def test_missing_token_counts_fall_back_to_estimates() -> None:
    body = {"message": {"role": "assistant", "content": "x" * 40}, "done": True}
    client, _ = make_client(lambda request: httpx.Response(200, json=body))
    result = await client.generate("m", MESSAGES)
    assert result.estimated_tokens is True
    assert result.tokens_out == 10
    assert result.tokens_in >= 1
    await client.aclose()


async def test_retries_on_503_then_succeeds() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, json=ok_body())

    client, sleeps = make_client(handler, max_retries=2)
    result = await client.generate("m", MESSAGES)
    assert result.text == "hi"
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]
    await client.aclose()


async def test_gives_up_after_max_retries() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500)

    client, sleeps = make_client(handler, max_retries=2)
    with pytest.raises(LLMUnavailableError):
        await client.generate("m", MESSAGES)
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]
    await client.aclose()


async def test_model_not_found_is_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"error": "model not found"})

    client, sleeps = make_client(handler)
    with pytest.raises(LLMModelNotFoundError, match="ollama pull"):
        await client.generate("ghost:1b", MESSAGES)
    assert len(calls) == 1 and sleeps == []
    await client.aclose()


async def test_client_error_is_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad request")

    client, _ = make_client(handler)
    with pytest.raises(LLMBadResponseError):
        await client.generate("m", MESSAGES)
    assert len(calls) == 1
    await client.aclose()


async def test_timeout_is_retried_then_raised() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("slow")

    client, _ = make_client(handler, max_retries=1)
    with pytest.raises(LLMTimeoutError):
        await client.generate("m", MESSAGES)
    assert len(calls) == 2
    await client.aclose()


async def test_connection_error_maps_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client, _ = make_client(handler, max_retries=0)
    with pytest.raises(LLMUnavailableError):
        await client.generate("m", MESSAGES)
    await client.aclose()


async def test_malformed_json_raises_bad_response() -> None:
    client, _ = make_client(lambda request: httpx.Response(200, text="not json"))
    with pytest.raises(LLMBadResponseError):
        await client.generate("m", MESSAGES)
    await client.aclose()


async def test_missing_message_field_raises_bad_response() -> None:
    client, _ = make_client(lambda request: httpx.Response(200, json={"done": True}))
    with pytest.raises(LLMBadResponseError):
        await client.generate("m", MESSAGES)
    await client.aclose()


async def test_invalid_requests_are_rejected() -> None:
    client, _ = make_client(lambda request: httpx.Response(200, json=ok_body()))
    with pytest.raises(ValueError):
        await client.generate("m", [])
    with pytest.raises(ValueError):
        await client.generate("m", MESSAGES, temperature=5.0)
    await client.aclose()


async def test_healthy_and_available_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200, json={"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:7b"}]}
        )

    client, _ = make_client(handler)
    assert await client.healthy() is True
    assert await client.available_models() == {"llama3.2:3b", "qwen2.5:7b"}
    await client.aclose()


async def test_unhealthy_when_server_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client, _ = make_client(handler)
    assert await client.healthy() is False
    with pytest.raises(LLMUnavailableError):
        await client.available_models()
    await client.aclose()


async def test_fake_client_records_calls_and_can_fail() -> None:
    fake = FakeLLMClient(reply="canned", tokens_out=7)
    result = await fake.generate("small-m", MESSAGES)
    assert result.text == "canned" and result.tokens_out == 7
    assert fake.calls[0][0] == "small-m"
    assert await fake.healthy() is True

    failing = FakeLLMClient(fail_with=LLMUnavailableError("down"))
    with pytest.raises(LLMUnavailableError):
        await failing.generate("m", MESSAGES)
    assert await failing.healthy() is False

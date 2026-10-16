"""Unit tests for the real provider adapters.

Model discovery is tested with the SDK clients mocked; chat is tested with a
deterministic fake LangChain chat model. Nothing here touches the network.
"""

from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from ollama import ResponseError

from app.core.config import Settings
from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderError
from app.providers.google import GoogleProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai_compat import OpenAICompatibleProvider, openai_model_filter
from app.providers.registry import build_providers
from app.schemas import ChatMessage
from tests.conftest import FakeChatModel

USER_HI = [ChatMessage(role="user", content="hi")]


def use_fake(monkeypatch, provider, fake: FakeChatModel) -> FakeChatModel:
    """Make `provider` chat through `fake` instead of a real LangChain integration."""
    monkeypatch.setattr(provider, "_build_model", lambda model, temperature: fake)
    return fake


async def collect(provider, messages=None) -> list[str]:
    return [t async for t in provider.stream_chat("m", messages or USER_HI)]


def status_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://example.test"))


def test_build_providers_orders_local_then_cloud_priority():
    settings = Settings(_env_file=None, cloud_priority=["grok", "openai"], grok_api_key="k")
    ids = [p.id for p in build_providers(settings)]
    assert ids[:3] == ["ollama", "grok", "openai"]
    assert set(ids) == {"ollama", "anthropic", "openai", "gemini", "grok", "meta"}
    configured = {p.id for p in build_providers(settings) if p.configured}
    assert configured == {"ollama", "grok"}


def test_openai_filter_hides_non_chat_models():
    f = openai_model_filter()
    assert f("gpt-5.5")
    assert not f("text-embedding-3-large")
    assert not f("whisper-1")
    assert not f("gpt-image-2")


async def test_ollama_lists_chat_models_and_skips_embeddings():
    provider = OllamaProvider("http://localhost:11434")

    def model(name, family="llama", families=None, size=1):
        details = SimpleNamespace(family=family, families=families or [family], parameter_size="3B")
        return SimpleNamespace(model=name, size=size, details=details)

    async def fake_list():
        return SimpleNamespace(
            models=[
                model("qwen3:8b"),
                model("llama3.2:latest"),
                model("nomic-embed-text:latest", "nomic-bert", ["nomic-bert"]),
            ]
        )

    provider._client.list = fake_list
    models = await provider.list_models()
    assert [m.name for m in models] == ["llama3.2", "qwen3:8b"]
    assert models[0].id == "llama3.2:latest"
    assert models[0].local is True


async def test_ollama_unreachable_raises_provider_error():
    provider = OllamaProvider("http://localhost:1")

    async def boom():
        raise ConnectionError("refused")

    provider._client.list = boom
    with pytest.raises(ProviderError, match="not reachable"):
        await provider.list_models()


async def test_ollama_streams_deltas_through_langchain(monkeypatch):
    provider = OllamaProvider("http://localhost:11434")
    use_fake(monkeypatch, provider, FakeChatModel(chunks=["Hel", "lo", ""]))
    assert await collect(provider) == ["Hel", "lo"]


async def test_only_text_blocks_are_streamed(monkeypatch):
    provider = OpenAICompatibleProvider("openai", "OpenAI", "k", "https://x")
    blocks = [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Hi"}]
    use_fake(monkeypatch, provider, FakeChatModel(chunks=[blocks, "!"]))
    assert await collect(provider) == ["Hi", "!"]


async def test_messages_are_converted_with_a_single_system_message(monkeypatch):
    provider = OllamaProvider("http://localhost:11434")
    fake = use_fake(monkeypatch, provider, FakeChatModel(chunks=["ok"]))
    messages = [
        ChatMessage(role="system", content="A"),
        ChatMessage(role="user", content="Use {braces} safely"),
        ChatMessage(role="assistant", content="Sure"),
        ChatMessage(role="system", content="B"),
        ChatMessage(role="user", content="Next"),
    ]
    await collect(provider, messages)

    sent = fake.seen[0]
    assert [type(m).__name__ for m in sent] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert sent[0].content == "A\n\nB"
    assert sent[1].content == "Use {braces} safely"  # never treated as a template


async def test_failure_after_first_chunk_is_reported_as_provider_error(monkeypatch):
    provider = OllamaProvider("http://localhost:11434")
    failing = FakeChatModel(chunks=["one", " two"], fail_with=httpx.ConnectError("x"), fail_after=1)
    use_fake(monkeypatch, provider, failing)

    received: list[str] = []
    with pytest.raises(ProviderError, match="not reachable"):
        async for text in provider.stream_chat("m", USER_HI):
            received.append(text)
    assert received == ["one"]


async def test_ollama_response_error_message_is_passed_on(monkeypatch):
    provider = OllamaProvider("http://localhost:11434")
    error = ResponseError("model 'x' not found", 404)
    use_fake(monkeypatch, provider, FakeChatModel(fail_with=error))
    with pytest.raises(ProviderError, match="Ollama error: model 'x' not found"):
        await collect(provider)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            openai.AuthenticationError("sk-secret", response=status_response(401), body=None),
            "invalid API key",
        ),
        (
            openai.RateLimitError("slow down", response=status_response(429), body=None),
            "rate limit or quota exceeded",
        ),
        (
            openai.NotFoundError("no such model", response=status_response(404), body=None),
            "model not found",
        ),
        (
            anthropic.AuthenticationError("bad key", response=status_response(401), body=None),
            "invalid API key",
        ),
        (httpx.ConnectError("refused"), "could not connect"),
        (httpx.ReadTimeout("too slow"), "timed out"),
        (ValueError("sk-secret leaked in a stack trace"), "request failed"),
    ],
)
async def test_failures_become_short_safe_provider_errors(monkeypatch, error, expected):
    provider = OpenAICompatibleProvider("openai", "OpenAI", "k", "https://x")
    use_fake(monkeypatch, provider, FakeChatModel(fail_with=error))
    with pytest.raises(ProviderError, match=expected) as info:
        await collect(provider)
    assert str(info.value).startswith("OpenAI: ")
    assert "sk-secret" not in str(info.value)


def test_ollama_builds_a_chat_ollama_model():
    model = OllamaProvider("http://localhost:11434")._build_model("llama3.2", 0.2)
    assert isinstance(model, ChatOllama)
    assert model.model == "llama3.2"
    assert model.temperature == 0.2


def test_anthropic_builds_a_chat_anthropic_model_and_clamps_temperature():
    model = AnthropicProvider("key")._build_model("claude-x", 1.5)
    assert isinstance(model, ChatAnthropic)
    assert model.temperature == 1.0


def test_openai_compatible_builds_chat_openai_with_configured_retries():
    settings = Settings(_env_file=None, openai_api_key="k", llm_max_retries=0)
    provider = next(p for p in build_providers(settings) if p.id == "openai")
    model = provider._build_model("gpt-x", None)
    assert isinstance(model, ChatOpenAI)
    assert model.max_retries == 0


def test_gemini_chats_through_the_google_integration():
    settings = Settings(_env_file=None, gemini_api_key="k")
    provider = next(p for p in build_providers(settings) if p.id == "gemini")
    assert isinstance(provider, GoogleProvider)
    assert isinstance(provider._build_model("gemini-x", 0.5), ChatGoogleGenerativeAI)


async def test_unconfigured_cloud_provider_raises():
    provider = OpenAICompatibleProvider("openai", "OpenAI", None, "https://x")
    assert provider.configured is False
    with pytest.raises(ProviderError, match="missing API key"):
        await provider.list_models()


@pytest.mark.parametrize(
    "provider",
    [
        AnthropicProvider(None),
        OpenAICompatibleProvider("openai", "OpenAI", None, "https://x"),
        GoogleProvider("gemini", "Google Gemini", None, "https://x"),
    ],
)
async def test_chat_without_a_key_fails_before_calling_the_model(provider):
    with pytest.raises(ProviderError, match="missing API key"):
        await collect(provider)

"""Cloud providers that speak the OpenAI Chat Completions protocol.

Chat goes through LangChain's ``ChatOpenAI`` pointed at each provider's
OpenAI-compatible endpoint: OpenAI, xAI Grok and Meta Llama. Gemini reuses the
model listing here but chats through its own adapter (``google.py``).

Model discovery uses the OpenAI SDK: LangChain has no model catalogue.
"""

from collections.abc import Callable

import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app.providers.base import LangChainProvider, ProviderError
from app.schemas import ModelInfo

ModelFilter = Callable[[str], bool]


def _chat_capable_openai(model_id: str) -> bool:
    blocked = (
        "embedding",
        "whisper",
        "tts",
        "dall-e",
        "image",
        "moderation",
        "audio",
        "realtime",
        "transcribe",
        "search",
        "davinci",
        "babbage",
    )
    return model_id.startswith(("gpt", "o", "chatgpt")) and not any(b in model_id for b in blocked)


def _chat_capable_generic(model_id: str) -> bool:
    blocked = ("embed", "image", "imagen", "veo", "tts", "audio", "vision-only", "aqa")
    return not any(b in model_id.lower() for b in blocked)


class OpenAICompatibleProvider(LangChainProvider):
    local = False

    def __init__(
        self,
        provider_id: str,
        label: str,
        api_key: str | None,
        base_url: str,
        *,
        default_model: str = "",
        timeout: float = 120.0,
        max_retries: int = 1,
        model_filter: ModelFilter = _chat_capable_generic,
    ) -> None:
        self.id = provider_id
        self.label = label
        self._api_key = api_key
        self._base_url = base_url
        self._default_model = default_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._filter = model_filter
        self._client = (
            AsyncOpenAI(
                api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
            )
            if api_key
            else None
        )

    @property
    def configured(self) -> bool:
        return self._client is not None

    @property
    def default_model(self) -> str:
        return self._default_model

    def _require_client(self) -> AsyncOpenAI:
        if self._client is None:
            raise ProviderError(f"{self.label} is not configured (missing API key)")
        return self._client

    async def list_models(self) -> list[ModelInfo]:
        client = self._require_client()
        try:
            page = await client.models.list()
            ids = [m.id.removeprefix("models/") async for m in page]
        except openai.APIError as exc:
            if self._default_model:  # listing unsupported: still offer the configured model
                ids = [self._default_model]
            else:
                raise ProviderError(f"{self.label}: could not list models ({exc})") from exc

        ids = sorted({i for i in ids if self._filter(i)} | ({self._default_model} - {""}))
        return [ModelInfo(id=i, name=i, provider=self.id, local=False) for i in ids]

    def _require_key(self) -> str:
        if not self._api_key:
            raise ProviderError(f"{self.label} is not configured (missing API key)")
        return self._api_key

    def _build_model(self, model: str, temperature: float | None) -> BaseChatModel:
        options = {} if temperature is None else {"temperature": temperature}
        return ChatOpenAI(
            model=model,
            api_key=self._require_key(),
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
            **options,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


def openai_model_filter() -> ModelFilter:
    return _chat_capable_openai

"""Anthropic Claude, chatted with through LangChain's ``ChatAnthropic``.

Model discovery uses the Anthropic SDK: LangChain has no model catalogue.
"""

import anthropic
from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

from app.providers.base import LangChainProvider, ProviderError
from app.schemas import ModelInfo

DEFAULT_MAX_TOKENS = 8192


class AnthropicProvider(LangChainProvider):
    id = "anthropic"
    label = "Anthropic Claude"
    local = False

    def __init__(
        self,
        api_key: str | None,
        default_model: str = "",
        timeout: float = 120.0,
        *,
        max_retries: int = 1,
    ):
        self._api_key = api_key
        self._default_model = default_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = (
            AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
            if api_key
            else None
        )

    @property
    def configured(self) -> bool:
        return self._client is not None

    @property
    def default_model(self) -> str:
        return self._default_model

    def _require_client(self) -> AsyncAnthropic:
        if self._client is None:
            raise ProviderError(f"{self.label} is not configured (missing API key)")
        return self._client

    async def list_models(self) -> list[ModelInfo]:
        client = self._require_client()
        try:
            models = [m async for m in client.models.list(limit=100)]
        except anthropic.APIError as exc:
            raise ProviderError(f"{self.label}: could not list models ({exc})") from exc
        return [
            ModelInfo(id=m.id, name=m.display_name or m.id, provider=self.id, local=False)
            for m in models
        ]

    def _build_model(self, model: str, temperature: float | None) -> BaseChatModel:
        if not self._api_key:
            raise ProviderError(f"{self.label} is not configured (missing API key)")
        options = {} if temperature is None else {"temperature": min(temperature, 1.0)}
        return ChatAnthropic(
            model=model,
            api_key=self._api_key,
            max_tokens=DEFAULT_MAX_TOKENS,
            timeout=self._timeout,
            max_retries=self._max_retries,
            **options,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

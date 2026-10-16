"""Local models served by Ollama, chatted with through LangChain's ``ChatOllama``.

Model discovery still uses the Ollama SDK: LangChain has no model catalogue.
"""

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from ollama import AsyncClient, ResponseError

from app.providers.base import LangChainProvider, ProviderError
from app.schemas import ModelInfo


class OllamaProvider(LangChainProvider):
    id = "ollama"
    label = "Ollama (local)"
    local = True

    def __init__(self, host: str, enabled: bool = True, timeout: float = 120.0) -> None:
        self._host = host
        self._enabled = enabled
        self._timeout = timeout
        self._client = AsyncClient(host=host, timeout=timeout)

    @property
    def configured(self) -> bool:
        return self._enabled and bool(self._host)

    async def list_models(self) -> list[ModelInfo]:
        try:
            response = await self._client.list()
        except (httpx.HTTPError, ConnectionError, ResponseError) as exc:
            raise ProviderError(f"Ollama is not reachable at {self._host}") from exc

        models: list[ModelInfo] = []
        for item in response.models:
            if not item.model:
                continue
            details = item.details
            # Embedding-only models cannot chat, so keep them out of the dropdown.
            families = (details.families or []) if details else []
            if any("bert" in f for f in families) or "embed" in item.model:
                continue
            models.append(
                ModelInfo(
                    id=item.model,
                    name=item.model.removesuffix(":latest"),
                    provider=self.id,
                    local=True,
                    size_bytes=item.size,
                    parameter_size=details.parameter_size if details else None,
                    family=details.family if details else None,
                )
            )
        return sorted(models, key=lambda m: m.name)

    def _build_model(self, model: str, temperature: float | None) -> BaseChatModel:
        options = {} if temperature is None else {"temperature": temperature}
        return ChatOllama(
            model=model,
            base_url=self._host,
            client_kwargs={"timeout": self._timeout},
            **options,
        )

    def _to_provider_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ResponseError):
            return ProviderError(f"Ollama error: {exc.error}")
        if isinstance(exc, httpx.HTTPError | ConnectionError):
            return ProviderError(f"Ollama is not reachable at {self._host}")
        return super()._to_provider_error(exc)

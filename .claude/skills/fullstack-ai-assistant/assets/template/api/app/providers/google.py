"""Google Gemini, chatted with through LangChain's ``ChatGoogleGenerativeAI``.

Model listing reuses the OpenAI-compatible endpoint from ``OpenAICompatibleProvider``
(Gemini serves one), while chat goes through Google's own LangChain integration.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from app.providers.openai_compat import OpenAICompatibleProvider


class GoogleProvider(OpenAICompatibleProvider):
    def _build_model(self, model: str, temperature: float | None) -> BaseChatModel:
        options = {} if temperature is None else {"temperature": temperature}
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=self._require_key(),
            timeout=self._timeout,
            max_retries=self._max_retries,
            **options,
        )

"""Provider interface shared by the local (Ollama) and cloud backends.

Chat calls go through LangChain chat models. LangChain and provider SDK types
stay inside ``app/providers``; the rest of the app only sees ``Provider``,
``ModelInfo``, ``ChatMessage`` and ``ProviderError``.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.schemas import ChatMessage, ModelInfo

logger = logging.getLogger(__name__)

DETAIL_MAX_CHARS = 200

# Messages are passed as message objects (not template strings), so user text
# containing braces is never interpreted as a template.
CHAT_PROMPT = ChatPromptTemplate.from_messages([MessagesPlaceholder("messages")])


class ProviderError(Exception):
    """A provider could not serve the request. Safe to show to the user."""


class Provider(ABC):
    id: str
    label: str
    local: bool = False

    @property
    @abstractmethod
    def configured(self) -> bool:
        """True when the provider has what it needs to run (host or API key)."""

    @property
    def default_model(self) -> str:
        """Preferred model id, if one is configured. Empty means 'first listed'."""
        return ""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return the models this provider can serve right now."""

    @abstractmethod
    def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas for a chat completion."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""


def to_langchain_messages(messages: list[ChatMessage]) -> list[BaseMessage]:
    """Convert app messages to LangChain messages.

    All system messages are merged into one leading ``SystemMessage``, which is
    the shape every provider accepts (Anthropic and Gemini require it).
    """
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    converted: list[BaseMessage] = [SystemMessage(content=system)] if system else []
    for m in messages:
        if m.role == "user":
            converted.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            converted.append(AIMessage(content=m.content))
    return converted


def chunk_text(chunk: Any) -> str:
    """Visible text of a streamed chunk.

    ``content`` is a string for most providers but a list of content blocks for
    some (e.g. Claude, Gemini). Only text blocks are shown; reasoning blocks are not.
    """
    content = chunk.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _status_code(exc: BaseException) -> int | None:
    candidates = (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    )
    return next((c for c in candidates if isinstance(c, int) and not isinstance(c, bool)), None)


def describe_failure(exc: BaseException) -> str:
    """Short, user-safe reason for a provider failure.

    Authentication, rate-limit, timeout and connection failures get fixed
    wording so keys, headers and stack traces can never reach the browser.
    """
    status = _status_code(exc)
    name = type(exc).__name__
    detail = str(getattr(exc, "message", "") or "")[:DETAIL_MAX_CHARS]
    suffix = f": {detail}" if detail else ""
    reason = "request failed"
    if status in (401, 403):
        reason = "invalid API key or access denied"
    elif status == 429:
        reason = "rate limit or quota exceeded"
    elif status in (400, 404, 422):
        reason = ("model not found" if status == 404 else "invalid request") + suffix
    elif "Timeout" in name or isinstance(exc, TimeoutError | httpx.TimeoutException):
        reason = "request timed out"
    elif "Connection" in name or isinstance(exc, ConnectionError | httpx.HTTPError):
        reason = "could not connect to the service"
    elif status is not None and status >= 500:
        reason = "the service had an error, try again later"
    return reason


class LangChainProvider(Provider):
    """A provider whose chat calls are served by a LangChain chat model."""

    @abstractmethod
    def _build_model(self, model: str, temperature: float | None) -> BaseChatModel:
        """Create the LangChain chat model. Raise ``ProviderError`` if not configured."""

    def _to_provider_error(self, exc: Exception) -> ProviderError:
        return ProviderError(f"{self.label}: {describe_failure(exc)}")

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        try:
            chain = CHAT_PROMPT | self._build_model(model, temperature)
            stream = chain.astream({"messages": to_langchain_messages(messages)})
            async with aclosing(stream):
                async for chunk in stream:
                    text = chunk_text(chunk)
                    if text:
                        yield text
        except ProviderError:
            raise
        except Exception as exc:
            logger.warning("%s/%s failed: %s", self.id, model, type(exc).__name__)
            raise self._to_provider_error(exc) from exc

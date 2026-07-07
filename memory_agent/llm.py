"""LLM client abstraction. Core package stays free of network dependencies;
`langchain_openai` is only imported lazily inside OpenAIClient.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], model: str | None = None) -> str:
        ...


class OpenAIClient:
    """Thin wrapper around `langchain_openai.ChatOpenAI`.

    `model` may carry an "openai:" prefix (matching the existing demo's
    env-var convention); the prefix is stripped before calling the API.

    `ChatOpenAI` binds its model name at construction time, but `complete()`
    accepts a per-call `model` override. To reconcile the two, chat model
    instances are cached in `self._chat_models`, keyed by the resolved
    (prefix-stripped) model name, and are only built the first time a given
    model name is used.

    For tests, pass `chat_model_factory` — a callable that takes a resolved
    model name and returns a `BaseChatModel`-like object (only `.invoke` is
    required) — to swap in a network-free fake instead of a real
    `ChatOpenAI`. This is the only supported dependency-injection seam; it
    keeps the constructor free of network calls when a factory is supplied.
    """

    def __init__(
        self,
        model: str,
        chat_model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model = model
        self._chat_model_factory = chat_model_factory or self._build_chat_model
        self._chat_models: dict[str, Any] = {}

    @staticmethod
    def _strip_prefix(model: str) -> str:
        if model.startswith("openai:"):
            return model[len("openai:"):]
        return model

    @staticmethod
    def _build_chat_model(model: str) -> Any:
        from langchain_openai import ChatOpenAI  # lazy import

        return ChatOpenAI(model=model)

    def _get_chat_model(self, model: str) -> Any:
        chat_model = self._chat_models.get(model)
        if chat_model is None:
            chat_model = self._chat_model_factory(model)
            self._chat_models[model] = chat_model
        return chat_model

    def complete(self, system: str, messages: list[dict], model: str | None = None) -> str:
        use_model = self._strip_prefix(model or self.model)
        chat_model = self._get_chat_model(use_model)
        full_messages = [{"role": "system", "content": system}] + list(messages)
        response = chat_model.invoke(full_messages)
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the assistant's text out of a chat model response.

        Prefers `AIMessage.text` (a property on langchain-core >= 1.0),
        which already normalizes both plain-string `content` and
        list-of-content-block `content`. Falls back to inspecting
        `.content` directly for lightweight test doubles that don't
        implement `.text`. Returns "" if no text is found.
        """
        text_accessor = getattr(response, "text", None)
        if text_accessor is not None:
            return str(text_accessor)

        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text") or "")
            return "".join(parts)

        return ""

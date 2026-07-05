"""LLM client abstraction. Core package stays free of network dependencies;
`openai` is only imported lazily inside OpenAIClient.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], model: str | None = None) -> str:
        ...


class OpenAIClient:
    """Thin wrapper around the OpenAI chat completions API.

    `model` may carry an "openai:" prefix (matching the existing demo's
    env-var convention); the prefix is stripped before calling the API.
    """

    def __init__(self, model: str) -> None:
        from openai import OpenAI  # lazy import

        self._client = OpenAI()
        self.model = model

    @staticmethod
    def _strip_prefix(model: str) -> str:
        if model.startswith("openai:"):
            return model[len("openai:"):]
        return model

    def complete(self, system: str, messages: list[dict], model: str | None = None) -> str:
        use_model = self._strip_prefix(model or self.model)
        full_messages = [{"role": "system", "content": system}] + list(messages)
        response = self._client.chat.completions.create(
            model=use_model,
            messages=full_messages,
        )
        return response.choices[0].message.content or ""

"""LLM client abstractions and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


@dataclass
class TokenLedger:
    """Accumulates approximate or provider-reported token usage by LLM role."""

    usage_by_role: dict[str, TokenUsage] = field(default_factory=dict)

    def ensure_roles(self, *roles: str) -> None:
        for role in roles:
            self.usage_by_role.setdefault(role, TokenUsage())

    def record(self, role: str, input_tokens: int, output_tokens: int) -> None:
        usage = self.usage_by_role.setdefault(role, TokenUsage())
        usage.input_tokens += max(0, int(input_tokens or 0))
        usage.output_tokens += max(0, int(output_tokens or 0))
        usage.calls += 1

    def record_text(self, role: str, input_text: str, output_text: str) -> None:
        self.record(role, estimate_tokens(input_text), estimate_tokens(output_text))

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            role: {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
                "calls": usage.calls,
            }
            for role, usage in sorted(self.usage_by_role.items())
        }


class LangChainTokenCallback(BaseCallbackHandler):
    """Record each provider-reported LangChain model call by role."""

    def __init__(self, token_ledger: TokenLedger, role: str) -> None:
        self.token_ledger = token_ledger
        self.role = role
        self.recorded_calls = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        input_tokens = 0
        output_tokens = 0
        found_usage = False
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage_metadata")
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens += int(
                usage.get("output_tokens") or usage.get("completion_tokens") or 0
            )
            found_usage = input_tokens > 0 or output_tokens > 0

        if not found_usage:
            for generation_list in getattr(response, "generations", []) or []:
                for generation in generation_list:
                    message = getattr(generation, "message", None)
                    metadata = getattr(message, "usage_metadata", None)
                    if not isinstance(metadata, dict):
                        continue
                    input_tokens += int(metadata.get("input_tokens") or 0)
                    output_tokens += int(metadata.get("output_tokens") or 0)
                    found_usage = True

        if found_usage:
            self.token_ledger.record(self.role, input_tokens, output_tokens)
            self.recorded_calls += 1


def estimate_tokens(text: str) -> int:
    """Small deterministic fallback when the provider omits usage metadata."""
    return max(1, (len(text or "") + 3) // 4) if text else 0


def response_token_usage(response: Any, prompt_text: str, output_text: str) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage")
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        if input_tokens is not None or output_tokens is not None:
            return int(input_tokens or 0), int(output_tokens or 0)
    return estimate_tokens(prompt_text), estimate_tokens(output_text)


class LLMClient(Protocol):
    """Minimal interface required by memory updater/session code."""

    def complete(self, system: str, messages: list[dict], model: str | None = None) -> str:
        ...


class OpenAIClient:
    """Thin wrapper around `langchain_openai.ChatOpenAI`.

    `model` may carry an "openai:" prefix matching the demo env-var
    convention; the prefix is stripped before calling the API. The class caches
    ChatOpenAI instances by resolved model name so per-call model overrides do
    not rebuild the same model repeatedly.
    """

    def __init__(
        self,
        model: str,
        chat_model_factory: Callable[[str], Any] | None = None,
        role: str | None = None,
        token_ledger: TokenLedger | None = None,
    ) -> None:
        self.model = model
        self.role = role
        self.token_ledger = token_ledger
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
        text = self._extract_text(response)
        if self.token_ledger is not None and self.role:
            prompt_text = "\n".join(str(message.get("content", "")) for message in full_messages)
            input_tokens, output_tokens = response_token_usage(response, prompt_text, text)
            self.token_ledger.record(self.role, input_tokens, output_tokens)
        return text

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the assistant text out of a chat model response."""
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

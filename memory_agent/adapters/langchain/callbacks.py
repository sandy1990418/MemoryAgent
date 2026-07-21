"""Optional LangChain callbacks kept outside the framework-free clients."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from memory_agent.clients.llm import TokenLedger


class LangChainTokenCallback(BaseCallbackHandler):
    """Record provider-reported token usage by role."""

    def __init__(self, token_ledger: TokenLedger, role: str) -> None:
        self.token_ledger = token_ledger
        self.role = role
        self.recorded_calls = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        input_tokens = output_tokens = 0
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage_metadata")
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        if not input_tokens and not output_tokens:
            for generations in getattr(response, "generations", []) or []:
                for generation in generations:
                    metadata = getattr(getattr(generation, "message", None), "usage_metadata", None)
                    if isinstance(metadata, dict):
                        input_tokens += int(metadata.get("input_tokens") or 0)
                        output_tokens += int(metadata.get("output_tokens") or 0)
        if input_tokens or output_tokens:
            self.token_ledger.record(self.role, input_tokens, output_tokens)
            self.recorded_calls += 1

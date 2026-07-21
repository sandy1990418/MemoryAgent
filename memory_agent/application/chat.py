"""Standalone chat/practical-memory application facade.

This module is intentionally small and dependency-light: it exposes the
practical structured-memory pieces without importing agent, BEAM, eval, or
mem0 integration code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from memory_agent.clients.llm import LLMClient, OpenAIClient, TokenLedger
from memory_agent.models.config import ProductMemoryConfig
from memory_agent.application.structured_service import StructuredMemoryService
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.update.compactor import MemoryCompactor
from memory_agent.update.updater import MemoryUpdater


@dataclass
class _RoleRecordingLLM:
    """Record estimated token usage per role for caller-supplied clients.

    OpenAIClient records provider-reported usage itself; this wrapper covers
    arbitrary LLMClient implementations with the deterministic estimator.
    """

    inner: LLMClient
    token_ledger: TokenLedger
    role: str

    def complete(self, system: str, messages: list[dict], model: str | None = None) -> str:
        output = self.inner.complete(system, messages, model)
        prompt = system + "\n" + "\n".join(
            str(message.get("content", "")) for message in messages
        )
        self.token_ledger.record_text(self.role, prompt, output)
        return output


@dataclass
class ChatMemory:
    """Practical chat memory package suitable for standalone handoff."""

    memory: Memory
    updater: MemoryUpdater
    compactor: MemoryCompactor | None = None
    compact_min_active_entries: int = 30
    token_ledger: TokenLedger | None = None
    service: StructuredMemoryService | None = None

    def __post_init__(self) -> None:
        if self.service is None:
            self.service = StructuredMemoryService(
                memory=self.memory,
                updater=self.updater,
                policy=self.memory.policy or self.updater.policy,
                compactor=self.compactor,
                compact_min_active_entries=self.compact_min_active_entries,
            )

    def token_usage(self) -> dict[str, dict[str, int]]:
        """Token spend per role ("updater"/"compactor") for this chat memory."""
        return self.token_ledger.to_dict() if self.token_ledger else {}

    def update(self, turns: list[Turn]) -> tuple[list[dict], list[dict]]:
        assert self.service is not None
        result = self.service.update(turns)
        if result.committed:
            return result.applied_ops, []
        if result.rejected_ops:
            return [], result.rejected_ops
        errors = result.verification.errors if result.verification else []
        return [], [{"op": None, "reason": result.failure_reason, "errors": errors}]

    def render(self, *, include_superseded: bool = False) -> str:
        return self.memory.render(include_superseded=include_superseded)


def build_chat_memory(
    llm: LLMClient | None = None,
    *,
    compact: bool = True,
    config: ProductMemoryConfig | None = None,
    config_path: str | Path = "configs/product.yaml",
) -> ChatMemory:
    """Build chat memory from product YAML/env without agent/eval/BEAM imports."""
    product = config or ProductMemoryConfig.from_yaml_env(config_path)
    ledger = TokenLedger()
    ledger.ensure_roles("updater", "compactor")
    if llm is None:
        updater_llm: LLMClient = OpenAIClient(
            product.memory_model, role="updater", token_ledger=ledger
        )
        compactor_llm: LLMClient = OpenAIClient(
            product.memory_model, role="compactor", token_ledger=ledger
        )
    else:
        updater_llm = _RoleRecordingLLM(llm, ledger, "updater")
        compactor_llm = _RoleRecordingLLM(llm, ledger, "compactor")
    # The production facade is deliberately chat-only.
    policy = CHAT_POLICY
    sections = list(CHAT_SECTIONS)
    memory = Memory(sections=sections, policy=policy)
    updater = MemoryUpdater(
        llm=updater_llm,
        sections=sections,
        policy=policy,
        update_memory_token_budget=product.update_memory_token_budget,
        evicted_turn_token_budget=product.evicted_turn_token_budget,
        max_candidate_entries=product.updater_max_candidate_entries,
    )
    compactor = (
        MemoryCompactor(
            llm=compactor_llm,
            sections=sections,
            policy=policy,
        )
        if compact
        else None
    )
    return ChatMemory(
        memory=memory,
        updater=updater,
        compactor=compactor,
        compact_min_active_entries=product.compaction_threshold,
        token_ledger=ledger,
    )

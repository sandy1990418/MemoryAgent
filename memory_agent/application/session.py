"""Framework-free structured-memory conversation session."""

from __future__ import annotations

import logging

from memory_agent.clients.llm import LLMClient
from memory_agent.application.structured_service import StructuredMemoryService
from memory_agent.core.sections import CHAT_SECTIONS, SectionConfig
from memory_agent.core.store import Memory
from memory_agent.core.transcript_store import Transcript
from memory_agent.core.window import WorkingWindow
from memory_agent.retrieval.selector import MemorySelector
from memory_agent.update.updater import MemoryUpdater

logger = logging.getLogger(__name__)


class MemorySession:
    """A single conversational session backed by structured living memory."""

    def __init__(
        self,
        chat_llm: LLMClient,
        updater: MemoryUpdater,
        sections: list[SectionConfig] = CHAT_SECTIONS,
        max_window_tokens: int = 2000,
        max_prompt_tokens: int | None = None,
        max_memory_tokens: int | None = None,
        base_system_prompt: str = "You are a helpful assistant.",
        memory_selector: MemorySelector | None = None,
    ) -> None:
        self.chat_llm = chat_llm
        self.updater = updater
        self.base_system_prompt = base_system_prompt
        self.max_prompt_tokens = max_prompt_tokens or max_window_tokens
        self.max_memory_tokens = (
            max_memory_tokens if max_memory_tokens is not None else self.max_prompt_tokens // 2
        )

        self.transcript = Transcript()
        self.window = WorkingWindow(max_tokens=max_window_tokens)
        self.memory = Memory(sections=sections, policy=updater.policy)
        self.service = StructuredMemoryService(
            memory=self.memory,
            updater=self.updater,
            policy=self.updater.policy,
        )
        self.memory_selector = memory_selector or MemorySelector(
            token_estimator=self.window.token_estimator,
            policy=updater.policy,
        )

        self.last_system_prompt: str = ""

    def _maybe_evict(self, query: str = "") -> None:
        prompt_overhead_tokens = self._prompt_overhead_tokens(query=query)
        if not self.window.needs_eviction(
            extra_tokens=prompt_overhead_tokens,
            max_tokens=self.max_prompt_tokens,
        ):
            return

        batch = self.window.eviction_batch(
            extra_tokens=prompt_overhead_tokens,
            max_tokens=self.max_prompt_tokens,
        )
        if not batch:
            return

        result = self.service.update(batch)

        if not result.committed:
            logger.warning(
                "Memory update was not committed (%s): %s",
                result.failure_reason,
                result.rejected_ops,
            )
            # Any rejection means some part of this batch may not have reached
            # memory. Keep the turns for retry instead of dropping context.
            return

        self.window.remove(batch)

    def _render_memory_for_prompt(self, query: str = "") -> str:
        selected_entries = self.memory_selector.select(
            memory=self.memory,
            query=query,
            max_tokens=self.max_memory_tokens,
        )
        return self.memory.render(
            max_tokens=self.max_memory_tokens,
            token_estimator=self.window.token_estimator,
            entries=selected_entries,
        )

    def _prompt_overhead_tokens(self, query: str = "") -> int:
        return self.window.token_estimator(self._build_system_prompt(query=query))

    def _build_system_prompt(self, query: str = "") -> str:
        # memory.render() already includes the narrative section when set.
        return f"{self.base_system_prompt}\n\n# Conversation Memory\n{self._render_memory_for_prompt(query=query)}"

    def send(self, user_text: str) -> str:
        user_turn = self.transcript.append("user", user_text)
        self.window.add(user_turn)

        self._maybe_evict(query=user_text)

        system = self._build_system_prompt(query=user_text)
        self.last_system_prompt = system

        messages = [{"role": t.role, "content": t.content} for t in self.window.turns()]
        reply = self.chat_llm.complete(system, messages)

        assistant_turn = self.transcript.append("assistant", reply)
        self.window.add(assistant_turn)

        return reply

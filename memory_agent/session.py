"""Ties together transcript, working window, memory, and the two LLMs."""

from __future__ import annotations

import logging

from memory_agent.llm import LLMClient
from memory_agent.memory import Memory
from memory_agent.sections import CHAT_SECTIONS, SectionConfig
from memory_agent.transcript import Transcript
from memory_agent.updater import MemoryUpdater, UpdateFailed
from memory_agent.window import WorkingWindow

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
    ) -> None:
        self.chat_llm = chat_llm
        self.updater = updater
        self.base_system_prompt = base_system_prompt
        self.max_prompt_tokens = max_prompt_tokens or max_window_tokens
        self.max_memory_tokens = (
            max_memory_tokens if max_memory_tokens is not None else self.max_prompt_tokens // 2
        )

        self.memory = Memory(sections=sections)
        self.transcript = Transcript()
        self.window = WorkingWindow(max_tokens=max_window_tokens)

        self.last_system_prompt: str = ""

    def _maybe_evict(self) -> None:
        prompt_overhead_tokens = self._prompt_overhead_tokens()
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

        try:
            _applied, rejected = self.updater.update(self.memory, batch)
        except UpdateFailed as exc:
            logger.warning("Memory update failed; keeping evicted turns for retry: %s", exc)
            return

        if rejected:
            logger.warning("Memory updater produced rejected ops: %s", rejected)
            # Any rejection means some part of this batch may not have reached
            # memory. Keep the turns for retry instead of dropping context.
            return

        self.window.remove(batch)

    def _render_memory_for_prompt(self) -> str:
        return self.memory.render(
            max_tokens=self.max_memory_tokens,
            token_estimator=self.window.token_estimator,
        )

    def _prompt_overhead_tokens(self) -> int:
        return self.window.token_estimator(self._build_system_prompt())

    def _build_system_prompt(self) -> str:
        # memory.render() already includes the narrative section when set.
        return f"{self.base_system_prompt}\n\n# Conversation Memory\n{self._render_memory_for_prompt()}"

    def send(self, user_text: str) -> str:
        user_turn = self.transcript.append("user", user_text)
        self.window.add(user_turn)

        self._maybe_evict()

        system = self._build_system_prompt()
        self.last_system_prompt = system

        messages = [{"role": t.role, "content": t.content} for t in self.window.turns()]
        reply = self.chat_llm.complete(system, messages)

        assistant_turn = self.transcript.append("assistant", reply)
        self.window.add(assistant_turn)

        return reply

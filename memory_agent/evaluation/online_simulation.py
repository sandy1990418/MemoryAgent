"""Production-like online replay with observable memory costs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import ceil
from typing import Callable, Iterable

from memory_agent.clients.llm import LLMClient
from memory_agent.structured.answer_context import (
    AnswerContextBudget,
    AnswerContextConfig,
    build_answer_memory_context,
)
from memory_agent.structured.compactor import MemoryCompactor
from memory_agent.structured.memory import Memory
from memory_agent.structured.quality import memory_quality_report
from memory_agent.structured.transcript import Transcript
from memory_agent.structured.updater import MemoryUpdater, UpdateFailed
from memory_agent.structured.window import WorkingWindow


class SimulationMode(str, Enum):
    TOKEN_ONLY = "token-only"
    LIVE = "live"


@dataclass(frozen=True)
class TranscriptExchange:
    user: str
    assistant: str


@dataclass(frozen=True)
class TurnSimulation:
    turn: int
    selected_ids: tuple[str, ...]
    injected_context: str
    injection_tokens: int
    working_window_tokens: int
    answer_input_tokens: int
    answer_called: bool
    memory_ids_before_turn: tuple[str, ...]
    memory_ids_after_turn: tuple[str, ...]


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, ceil(percentile * len(ordered)) - 1)]


class OnlineSimulation:
    """Replay turns using the production answer-context path.

    Ordering is deliberately explicit: user append, pre-turn select/inject,
    optional answer, assistant append, and only then eviction/update.
    """

    def __init__(
        self,
        *,
        memory: Memory,
        updater: MemoryUpdater,
        answer_context_config: AnswerContextConfig,
        answer_memory_budget: int | None,
        max_window_tokens: int,
        mode: SimulationMode | str = SimulationMode.TOKEN_ONLY,
        answer_llm: LLMClient | None = None,
        base_system_prompt: str = "You are a helpful assistant.",
        token_estimator: Callable[[str], int] | None = None,
        compactor: MemoryCompactor | None = None,
    ) -> None:
        self.mode = SimulationMode(mode)
        if self.mode is SimulationMode.LIVE and answer_llm is None:
            raise ValueError("live simulation requires an explicit answer_llm")
        self.memory = memory
        self.updater = updater
        self.answer_context_config = answer_context_config
        self.answer_memory_budget = answer_memory_budget
        self.answer_llm = answer_llm
        self.base_system_prompt = base_system_prompt
        self.estimator_policy = (
            "caller_supplied" if token_estimator is not None else "characters_divided_by_four"
        )
        self.token_estimator = token_estimator or (lambda text: max(1, len(text) // 4) if text else 0)
        self.window = WorkingWindow(max_window_tokens, token_estimator=self.token_estimator)
        self.transcript = Transcript()
        self.compactor = compactor
        self.turns: list[TurnSimulation] = []
        self.answer_calls = 0

    def _evict_and_update(self) -> None:
        if not self.window.needs_eviction():
            return
        batch = self.window.eviction_batch()
        if not batch:
            return
        try:
            _applied, rejected = self.updater.update(self.memory, batch)
        except UpdateFailed:
            return
        if rejected:
            return
        self.window.remove(batch)
        if self.compactor is not None:
            candidates = self.compactor.detect_candidates(self.memory)
            self.compactor.compact_candidates(self.memory, candidates)

    def run(self, exchanges: Iterable[TranscriptExchange]) -> dict[str, object]:
        for index, exchange in enumerate(exchanges, start=1):
            user_turn = self.transcript.append("user", exchange.user)
            self.window.add(user_turn)

            before_ids = tuple(sorted(self.memory.entries))
            context = build_answer_memory_context(
                query=exchange.user,
                memory=self.memory,
                config=self.answer_context_config,
                budget=AnswerContextBudget(self.answer_memory_budget),
            )
            injection_tokens = self.token_estimator(context.rendered_context)
            working_messages = [
                {"role": turn.role, "content": turn.content} for turn in self.window.turns()
            ]
            working_window_text = "\n".join(
                f"{message['role']}: {message['content']}" for message in working_messages
            )
            working_window_tokens = self.token_estimator(working_window_text)
            system = f"{self.base_system_prompt}\n\n# Conversation Memory\n{context.rendered_context}"
            answer_input_tokens = self.token_estimator(system) + working_window_tokens

            answer_called = self.mode is SimulationMode.LIVE
            if answer_called:
                self.answer_calls += 1
                assert self.answer_llm is not None
                assistant_text = self.answer_llm.complete(system, working_messages)
            else:
                assistant_text = exchange.assistant

            assistant_turn = self.transcript.append("assistant", assistant_text)
            self.window.add(assistant_turn)
            self._evict_and_update()
            self.turns.append(TurnSimulation(
                turn=index,
                selected_ids=context.selected_ids,
                injected_context=context.rendered_context,
                injection_tokens=injection_tokens,
                working_window_tokens=working_window_tokens,
                answer_input_tokens=answer_input_tokens,
                answer_called=answer_called,
                memory_ids_before_turn=before_ids,
                memory_ids_after_turn=tuple(sorted(self.memory.entries)),
            ))
        return self.report()

    def report(self) -> dict[str, object]:
        injections = [turn.injection_tokens for turn in self.turns]
        answer_inputs = [turn.answer_input_tokens for turn in self.turns]
        total = sum(injections)
        quality = asdict(memory_quality_report(self.memory))
        compactor = asdict(self.compactor.metrics) if self.compactor is not None else None
        return {
            "mode": self.mode.value,
            "turn_count": len(self.turns),
            "answer_calls": self.answer_calls,
            "injection": {
                "source": "estimator",
                "estimator_policy": self.estimator_policy,
                "average_tokens": total / len(injections) if injections else 0.0,
                "p50_tokens": _percentile(injections, 0.50),
                "p95_tokens": _percentile(injections, 0.95),
                "max_tokens": max(injections, default=0),
                "cumulative_tokens": total,
                "zero_injection_turns": sum(value == 0 for value in injections),
            },
            "answer_input": {
                "source": "estimator",
                "estimator_policy": self.estimator_policy,
                "average_tokens": sum(answer_inputs) / len(answer_inputs) if answer_inputs else 0.0,
                "p50_tokens": _percentile(answer_inputs, 0.50),
                "p95_tokens": _percentile(answer_inputs, 0.95),
                "max_tokens": max(answer_inputs, default=0),
                "cumulative_tokens": sum(answer_inputs),
            },
            "updater": self.updater.update_token_usage(),
            "compactor": compactor,
            "quality": quality,
            "turns": [asdict(turn) for turn in self.turns],
        }

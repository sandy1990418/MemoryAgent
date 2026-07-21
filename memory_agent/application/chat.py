"""Framework-neutral product facade for structured chat memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from memory_agent.clients.llm import LLMClient, OpenAIClient, TokenLedger
from memory_agent.models.config import ProductMemoryConfig
from memory_agent.application.structured_service import StructuredMemoryService
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.retrieval.selector import MemorySelector
from memory_agent.update.compactor import MemoryCompactor
from memory_agent.update.updater import MemoryUpdater


def _validate_token_budget(value: int, *, name: str) -> int:
    """Return a usable positive token budget or fail at the API boundary."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


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
    """Production chat memory package suitable for standalone handoff."""

    memory: Memory
    updater: MemoryUpdater
    compactor: MemoryCompactor | None = None
    compact_min_active_entries: int = 30
    answer_memory_token_budget: int = 600
    token_ledger: TokenLedger | None = None
    service: StructuredMemoryService | None = None
    memory_selector: MemorySelector | None = None
    _committed_turn_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _deferred_turns: list[Turn] = field(default_factory=list, init=False, repr=False)
    _last_update_diagnostics: dict[str, object] = field(
        default_factory=lambda: {
            "planned_turn_ids": [],
            "submitted_turn_ids": [],
            "committed_turn_ids": [],
            "deferred_turn_ids": [],
            "retained_deferred_turn_ids": [],
            "dropped_turn_ids": [],
            "attempt_count": 0,
            "status": "idle",
        },
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.answer_memory_token_budget = _validate_token_budget(
            self.answer_memory_token_budget,
            name="answer_memory_token_budget",
        )
        if self.service is None:
            self.service = StructuredMemoryService(
                memory=self.memory,
                updater=self.updater,
                policy=self.memory.policy or self.updater.policy,
                compactor=self.compactor,
                compact_min_active_entries=self.compact_min_active_entries,
            )
        if self.memory_selector is None:
            self.memory_selector = MemorySelector(policy=self.memory.policy)

    def token_usage(self) -> dict[str, dict[str, int]]:
        """Token spend per role ("updater"/"compactor") for this chat memory."""
        return self.token_ledger.to_dict() if self.token_ledger else {}

    def update_diagnostics(self) -> dict[str, object]:
        """Return structural metadata for the latest public update call."""
        return {
            key: list(value) if isinstance(value, list) else value
            for key, value in self._last_update_diagnostics.items()
        }

    def update(self, turns: list[Turn]) -> tuple[list[dict], list[dict]]:
        """Update uncommitted turns, retaining only a deferred suffix.

        The service may commit a verified leading prefix while deferring a
        later microbatch.  Record that prefix and retain the suffix locally so
        a caller can safely retry the same list (or submit the next batch),
        but do not retry the suffix inside this call: the updater has already
        exhausted its per-batch retry budget.  Keeping retry ownership at the
        public-call boundary avoids replaying committed work and multiplying
        failed-batch LLM calls without dropping turns when an evaluation caller
        does not inspect the return value.
        """
        assert self.service is not None
        pending: list[Turn] = []
        seen_ids: set[int] = set()
        for turn in [*self._deferred_turns, *turns]:
            if turn.id in self._committed_turn_ids or turn.id in seen_ids:
                continue
            seen_ids.add(turn.id)
            pending.append(turn)
        self._deferred_turns.clear()
        if not pending:
            self._last_update_diagnostics = {
                "planned_turn_ids": [],
                "submitted_turn_ids": [],
                "committed_turn_ids": [],
                "deferred_turn_ids": [],
                "retained_deferred_turn_ids": [],
                "dropped_turn_ids": [],
                "attempt_count": 0,
                "status": "idempotent",
            }
            return [], []

        result = self.service.update(pending)
        pending_ids = {turn.id for turn in pending}
        raw_committed_ids = result.diagnostics.get("committed_turn_ids", [])
        committed_ids = (
            {
                turn_id
                for turn_id in raw_committed_ids
                if isinstance(turn_id, int) and turn_id in pending_ids
            }
            if isinstance(raw_committed_ids, list)
            else set()
        )
        self._committed_turn_ids.update(committed_ids)
        self._deferred_turns = [
            turn for turn in pending if turn.id not in committed_ids
        ]
        self._last_update_diagnostics = {
            key: list(value) if isinstance(value, list) else value
            for key, value in result.diagnostics.items()
        }
        self._last_update_diagnostics.update(
            {
                "planned_turn_ids": [turn.id for turn in pending],
                "submitted_turn_ids": [turn.id for turn in pending],
                "committed_turn_ids": [turn.id for turn in pending if turn.id in committed_ids],
                "deferred_turn_ids": [
                    turn.id for turn in pending if turn.id not in committed_ids
                ],
                "retained_deferred_turn_ids": [
                    turn.id for turn in self._deferred_turns
                ],
                "dropped_turn_ids": [],
                "attempt_count": 1,
            }
        )

        if result.rejected_ops:
            return result.applied_ops, result.rejected_ops
        if result.failure_reason:
            errors = result.verification.errors if result.verification else []
            return result.applied_ops, [
                {"op": None, "reason": result.failure_reason, "errors": errors}
            ]
        return result.applied_ops, []

    def render(
        self,
        *,
        include_superseded: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Render bounded answer memory, optionally widening this call only.

        The configured ``answer_memory_token_budget`` remains the default. A
        caller can request a larger (or smaller) bounded context for a
        particular answer without rebuilding the chat object. Selection stays
        structural and query-independent; the same effective budget is used
        for selection and final rendering.
        """
        assert self.memory_selector is not None
        budget = self.answer_memory_token_budget if max_tokens is None else _validate_token_budget(
            max_tokens,
            name="max_tokens",
        )
        entries = self.memory_selector.select(
            memory=self.memory,
            max_tokens=budget,
            include_superseded=include_superseded,
        )
        return self.memory.render(
            entries=entries,
            include_superseded=include_superseded,
            max_tokens=budget,
        )


def build_chat_memory(
    llm: LLMClient | None = None,
    *,
    compact: bool = True,
    config: ProductMemoryConfig | None = None,
    config_path: str | Path = "configs/product.yaml",
) -> ChatMemory:
    """Build chat memory from the product configuration."""
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
        answer_memory_token_budget=product.answer_memory_token_budget,
        token_ledger=ledger,
    )

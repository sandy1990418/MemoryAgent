"""Update pipeline that turns evicted turns into memory operations."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import mean, median
from typing import Callable

from memory_agent.clients.llm import LLMClient
from memory_agent.core.sections import SectionConfig
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import (
    CHAT_POLICY,
    StructuredMemoryPolicy,
    validate_policy_sections,
)
from memory_agent.update.operations import UpdateFailed, parse_memory_ops
from memory_agent.update.prompts import build_updater_prompt
from memory_agent.update.selector import UpdateMemorySelector

logger = logging.getLogger(__name__)


def _debug_ops(label: str, ops: list[dict]) -> None:
    """Log op volume by kind and section; the fastest way to find where an
    entry explosion (V1~V71-style) is coming from. Enable via DEBUG logging."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    dict_ops = [op for op in ops if isinstance(op, dict)]
    logger.debug(
        "%s count=%d by_op=%s by_section=%s",
        label,
        len(ops),
        dict(Counter(op.get("op") for op in dict_ops)),
        dict(Counter(op.get("section") for op in dict_ops if op.get("section"))),
    )

def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class UpdateTokenReport:
    call_index: int
    evicted_turn_count: int
    visible_memory_entry_count: int
    visible_entries_by_section: dict[str, int]
    estimator_policy: str
    calls: int
    system_tokens: int
    visible_memory_tokens: int
    evicted_turn_tokens: int
    output_tokens: int
    retry_tokens: int = 0
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    llm_ops_count: int = 0
    rejected_ops_count: int = 0
    llm_call_required_reason: str = "call:possible_durable_assertion"
    required_overflow_tokens: int = 0
    mandatory_overflow_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def input_tokens(self) -> int:
        return self.system_tokens + self.visible_memory_tokens + self.evicted_turn_tokens + self.retry_tokens


@dataclass(frozen=True)
class TurnGroup:
    turns: tuple[Turn, ...]
    group_type: str
    mandatory: bool = False


@dataclass
class PreparedBatch:
    """One staged updater batch and the turns it is responsible for."""

    turns: tuple[Turn, ...]
    applied_ops: list[dict] = field(default_factory=list)
    rejected_ops: list[dict] = field(default_factory=list)


@dataclass
class PreparedUpdate:
    trial_memory: Memory
    applied_ops: list[dict]
    rejected_ops: list[dict]
    base_revision: int
    _committed: bool = False
    batches: list[PreparedBatch] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def commit(self, live_memory: Memory) -> None:
        if self.rejected_ops:
            raise RuntimeError("cannot commit a prepared update with rejected operations")
        if self._committed:
            raise RuntimeError("prepared update was already committed")
        live_memory.commit_trial(self.trial_memory, self.base_revision)
        self._committed = True

class MemoryUpdater:
    """Asks an LLM to translate evicted turns into ADD/UPDATE/SUPERSEDE ops."""

    def __init__(
        self,
        llm: LLMClient,
        sections: list[SectionConfig],
        model: str | None = None,
        max_memory_tokens: int | None = None,
        token_estimator: Callable[[str], int] | None = None,
        max_retries: int = 1,
        update_context_max_entries: int = 40,
        update_memory_token_budget: int | None = None,
        evicted_turn_token_budget: int | None = None,
        policy: StructuredMemoryPolicy | None = None,
        max_candidate_entries: int = 8,
    ) -> None:
        self.llm = llm
        self.sections = sections
        self.model = model
        self.max_memory_tokens = max_memory_tokens
        self.token_estimator = token_estimator or _default_token_estimator
        self.max_retries = max(0, max_retries)
        self.update_context_max_entries = update_context_max_entries
        self.update_memory_token_budget = (
            max_memory_tokens if update_memory_token_budget is None else update_memory_token_budget
        )
        self.evicted_turn_token_budget = evicted_turn_token_budget
        self.token_reports: list[UpdateTokenReport] = []
        # Every updater uses the one chat retention contract.
        self.policy = policy or CHAT_POLICY
        self.max_candidate_entries = max_candidate_entries
        self.decision_reasons: Counter[str] = Counter()
        self.evicted_user_assistant_pairs = 0
        self.turn_selection_reports: list[dict] = []
        self._last_update_diagnostics: dict[str, object] = {
            "planned_turn_ids": [],
            "planned_batch_turn_ids": [],
            "committed_turn_ids": [],
            "deferred_turn_ids": [],
            "dropped_turn_ids": [],
            "status": "idle",
        }
        # Fail fast on section mismatches instead of silently running
        # with retention behavior the caller did not intend.
        validate_policy_sections(self.policy, sections)

    def update_token_usage(self) -> dict:
        """Return per-call and aggregate estimated/provider attribution."""
        calls = sum(report.calls for report in self.token_reports)
        totals = {
            name: sum(getattr(report, name) for report in self.token_reports)
            for name in (
                "system_tokens", "visible_memory_tokens", "evicted_turn_tokens", "output_tokens"
            )
        }
        total_tokens = sum(totals.values())
        inputs = [r.system_tokens + r.visible_memory_tokens + r.evicted_turn_tokens + r.retry_tokens for r in self.token_reports]
        provider_inputs = [r.provider_input_tokens for r in self.token_reports if r.provider_input_tokens is not None]
        provider_outputs = [r.provider_output_tokens for r in self.token_reports if r.provider_output_tokens is not None]
        visible_by_section: Counter[str] = Counter()
        for report in self.token_reports:
            visible_by_section.update(report.visible_entries_by_section)
        p95 = 0
        if inputs:
            inputs = sorted(inputs)
            p95 = inputs[math.ceil(len(inputs) * .95) - 1]
        return {
            "source": "estimator",
            "estimator_policy": "characters_divided_by_four",
            "calls": calls,
            **totals,
            "total_tokens": total_tokens,
            "average_tokens_per_call": total_tokens / calls if calls else 0.0,
            "mean_input_tokens_per_call": mean(inputs) if inputs else 0.0,
            "median_input_tokens_per_call": median(inputs) if inputs else 0.0,
            "p95_input_tokens_per_call": p95,
            "updater_calls_per_evicted_pair": calls / self.evicted_user_assistant_pairs if self.evicted_user_assistant_pairs else 0.0,
            "evicted_user_assistant_pairs": self.evicted_user_assistant_pairs,
            "decision_reasons": dict(self.decision_reasons),
            "retries": sum(1 for r in self.token_reports if r.retry_tokens),
            "retry_tokens": sum(r.retry_tokens for r in self.token_reports),
            "rejected_ops_count": sum(r.rejected_ops_count for r in self.token_reports),
            "mandatory_turn_budget_overflow_count": sum(
                bool(report["mandatory_overflow_tokens"])
                for report in self.turn_selection_reports
            ),
            "dropped_turn_count": sum(
                len(report["dropped_turn_ids"]) for report in self.turn_selection_reports
            ),
            "deferred_turn_count": sum(
                len(report.get("deferred_turn_ids", [])) for report in self.turn_selection_reports
            ),
            "non_contiguous_selection_count": sum(
                not report["selection_is_contiguous"] for report in self.turn_selection_reports
            ),
            "turn_selection_reports": list(self.turn_selection_reports),
            "visible_entries_by_section": dict(visible_by_section),
            "provider_reported_input_tokens": sum(provider_inputs),
            "provider_reported_output_tokens": sum(provider_outputs),
            "provider_reported_calls": len(provider_inputs),
            "calls_detail": [
                {**asdict(report), "system_schema_tokens": report.system_tokens,
                 "input_tokens": report.input_tokens}
                for report in self.token_reports
            ],
            "last_update_diagnostics": self.update_diagnostics(),
        }

    def _turns_within_budget(self, turns: list[Turn]) -> list[Turn]:
        """Return the historical newest bounded view for diagnostics callers.

        Production updates no longer call this compatibility helper; they use
        :meth:`_plan_turn_batches` and stage every turn. Retaining this direct
        helper avoids breaking offline callers that used it only to inspect a
        single bounded prompt view.
        """
        groups = self._turn_groups(turns)
        prompt_turns_by_group = {
            id(group): self._prompt_turns_for_group(group) for group in groups
        }
        budget = self.evicted_turn_token_budget
        selected: list[TurnGroup] = []
        used = 0
        if budget is None:
            selected = groups
        else:
            for group in reversed(groups):
                tokens = sum(
                    self.token_estimator(turn.content)
                    for turn in prompt_turns_by_group[id(group)]
                )
                if group.mandatory or used + tokens <= budget:
                    selected.append(group)
                    used += tokens
                else:
                    break
            selected.reverse()
        selected_ids = {
            turn.id
            for group in selected
            for turn in prompt_turns_by_group[id(group)]
        }
        selected_tokens = sum(
            self.token_estimator(turn.content) for turn in turns if turn.id in selected_ids
        )
        overflow = max(0, selected_tokens - budget) if budget is not None else 0
        dropped = [turn for turn in turns if turn.id not in selected_ids]
        self.turn_selection_reports.append({
            "selected_turn_ids": [turn.id for turn in turns if turn.id in selected_ids],
            "dropped_turn_ids": [turn.id for turn in dropped],
            "selected_group_count": len(selected),
            "dropped_group_count": len(groups) - len(selected),
            "selected_turn_tokens": selected_tokens,
            "dropped_turn_tokens": sum(self.token_estimator(turn.content) for turn in dropped),
            "mandatory_overflow_tokens": overflow,
            "oversized_mandatory_group": overflow > 0,
            "selection_is_contiguous": not dropped or not selected or max(t.id for t in dropped) < min(selected_ids),
            "groups": [
                {
                    "type": group.group_type,
                    "turn_ids": [turn.id for turn in group.turns],
                    "mandatory": group.mandatory,
                }
                for group in groups
            ],
        })
        return [turn for turn in turns if turn.id in selected_ids]

    def _plan_turn_batches(self, turns: list[Turn]) -> list[list[Turn]]:
        """Plan oldest-first, contiguous, complete conversational batches.

        A budget limits each LLM prompt, not the set of turns eligible for a
        committed update. An oversized exchange remains intact as one batch;
        the prompt builder provides the existing content-level bounding.
        """
        groups = self._turn_groups(turns)
        budget = self.evicted_turn_token_budget
        batches: list[list[Turn]] = []
        current: list[Turn] = []
        used = 0
        for group in groups:
            group_turns = list(self._prompt_turns_for_group(group))
            group_tokens = sum(self.token_estimator(turn.content) for turn in group_turns)
            if current and budget is not None and used + group_tokens > budget:
                batches.append(current)
                current = []
                used = 0
            current.extend(group_turns)
            used += group_tokens
        if current:
            batches.append(current)

        planned_ids = [turn.id for turn in turns]
        planned_batch_ids = [[turn.id for turn in batch] for batch in batches]
        oversized_overflow = 0
        if budget is not None:
            oversized_overflow = sum(
                max(
                    0,
                    sum(self.token_estimator(turn.content) for turn in group.turns) - budget,
                )
                for group in groups
            )
        self.turn_selection_reports.append({
            "selected_turn_ids": planned_ids,
            "planned_turn_ids": planned_ids,
            "planned_batch_turn_ids": planned_batch_ids,
            "dropped_turn_ids": [],
            "deferred_turn_ids": [],
            "selected_group_count": len(groups),
            "dropped_group_count": 0,
            "selected_turn_tokens": sum(self.token_estimator(turn.content) for turn in turns),
            "dropped_turn_tokens": 0,
            "mandatory_overflow_tokens": oversized_overflow,
            "oversized_mandatory_group": oversized_overflow > 0,
            "selection_is_contiguous": True,
            "groups": [
                {
                    "type": group.group_type,
                    "turn_ids": [turn.id for turn in group.turns],
                    "mandatory": group.mandatory,
                }
                for group in groups
            ],
        })
        return batches

    def update_diagnostics(self) -> dict[str, object]:
        """Return diagnostics for the most recent atomic update attempt."""
        return {
            key: list(value) if isinstance(value, list) else value
            for key, value in self._last_update_diagnostics.items()
        }

    def _set_update_diagnostics(
        self,
        *,
        turns: list[Turn],
        batches: list[PreparedBatch] | list[list[Turn]],
        status: str,
        committed: bool = False,
    ) -> dict[str, object]:
        batch_turn_ids = [
            [turn.id for turn in batch.turns] if isinstance(batch, PreparedBatch)
            else [turn.id for turn in batch]
            for batch in batches
        ]
        planned_ids = [turn.id for turn in turns]
        committed_ids = planned_ids if committed else []
        deferred_ids = [] if committed else planned_ids
        diagnostics: dict[str, object] = {
            "planned_turn_ids": planned_ids,
            "planned_batch_turn_ids": batch_turn_ids,
            "committed_turn_ids": committed_ids,
            "deferred_turn_ids": deferred_ids,
            "dropped_turn_ids": [],
            "status": status,
        }
        self._last_update_diagnostics = diagnostics
        return diagnostics

    @staticmethod
    def _prompt_turns_for_group(group: TurnGroup) -> tuple[Turn, ...]:
        """Return the complete conversational group for LLM extraction.

        The updater does not infer ownership or durability from turn roles;
        those are model decisions.  Keeping groups intact also preserves the
        provenance context required by the structural validator.
        """
        return group.turns

    @staticmethod
    def _turn_groups(turns: list[Turn]) -> list[TurnGroup]:
        """Group contiguous turns without interpreting their content."""
        raw: list[tuple[list[Turn], str]] = []
        current: list[Turn] = []
        for turn in turns:
            if turn.role == "user" and current:
                raw.append((current, MemoryUpdater._group_type(current)))
                current = []
            current.append(turn)
        if current:
            raw.append((current, MemoryUpdater._group_type(current)))

        return [
            TurnGroup(tuple(group), kind, mandatory=index == len(raw) - 1)
            for index, (group, kind) in enumerate(raw)
        ]

    @staticmethod
    def _group_type(turns: list[Turn]) -> str:
        roles = {turn.role for turn in turns}
        if turns and turns[0].role == "user" and len(turns) == 1:
            return "unresolved_user"
        if turns and turns[0].role == "user" and "assistant" in roles:
            return "user_assistant"
        return "standalone"

    def _select_update_context_entries(self, memory: Memory, evicted_turns: list[Turn]) -> list:
        """Pick a bounded recency view without semantic matching.

        Visibility is an operational token-budget concern.  The model sees
        recent active entries first, followed by superseded history; section
        names and turn words do not influence retention decisions.
        """
        entries = sorted(
            memory.entries.values(),
            key=lambda entry: (
                entry.status != "active",
                -(max(entry.provenance) if entry.provenance else -1),
                entry.id,
            ),
        )
        return entries[: self.update_context_max_entries]

    def _build_prompt(
        self,
        memory: Memory,
        evicted_turns: list[Turn],
        visible_entries: list | tuple | None = None,
    ) -> tuple[str, list[dict]]:
        if visible_entries is None:
            # Preserve the direct helper's historical diagnostic behavior;
            # production ``update`` always supplies its explicit visible set.
            visible_entries = (
                tuple(memory.entries.values())
                if len(memory.entries) <= self.update_context_max_entries
                else tuple(self._select_update_context_entries(memory, evicted_turns))
            )
        current_memory = memory.render(
            include_superseded=True,
            entries=visible_entries,
        ) or "(No memory entries yet.)"

        return build_updater_prompt(
            sections=self.sections,
            policy=self.policy,
            current_memory=current_memory,
            turns=evicted_turns,
        )

    def _provider_usage(self) -> tuple[int, int, int] | None:
        ledger = getattr(self.llm, "token_ledger", None)
        role = getattr(self.llm, "role", None)
        usage = getattr(ledger, "usage_by_role", {}).get(role) if ledger is not None and role else None
        if usage is None:
            return None
        return usage.calls, usage.input_tokens, usage.output_tokens

    def update(self, memory: Memory, evicted_turns: list[Turn]) -> tuple[list[dict], list[dict]]:
        result = self.prepare_update(memory, evicted_turns)
        if not result.rejected_ops:
            result.commit(memory)
            self._mark_update_committed(result)
        return result.applied_ops, result.rejected_ops

    def prepare_update(self, memory: Memory, evicted_turns: list[Turn]) -> PreparedUpdate:
        turns = list(evicted_turns)
        if not turns:
            self.decision_reasons["skip:empty_batch"] += 1
        batches = self._plan_turn_batches(turns)
        trial, base_revision = memory.transaction_snapshot()
        staged_batches: list[PreparedBatch] = []
        all_applied: list[dict] = []
        try:
            for batch in batches:
                applied, rejected = self._update_trial(trial, batch)
                prepared_batch = PreparedBatch(tuple(batch), applied, rejected)
                staged_batches.append(prepared_batch)
                if rejected:
                    diagnostics = self._set_update_diagnostics(
                        turns=turns,
                        batches=batches,
                        status="rejected",
                    )
                    return PreparedUpdate(
                        trial,
                        [],
                        rejected,
                        base_revision,
                        batches=staged_batches,
                        diagnostics=diagnostics,
                    )
                all_applied.extend(applied)
        except Exception:
            self._set_update_diagnostics(
                turns=turns,
                batches=batches,
                status="failed",
            )
            raise

        diagnostics = self._set_update_diagnostics(
            turns=turns,
            batches=staged_batches,
            status="prepared",
        )
        return PreparedUpdate(
            trial,
            all_applied,
            [],
            base_revision,
            batches=staged_batches,
            diagnostics=diagnostics,
        )

    def _mark_update_committed(self, prepared: PreparedUpdate) -> None:
        """Finalize diagnostics after the outer live-memory commit."""
        turns = [turn for batch in prepared.batches for turn in batch.turns]
        diagnostics = self._set_update_diagnostics(
            turns=turns,
            batches=prepared.batches,
            status="committed",
            committed=True,
        )
        prepared.diagnostics.clear()
        prepared.diagnostics.update(diagnostics)

    def _update_trial(self, memory: Memory, evicted_turns: list[Turn]) -> tuple[list[dict], list[dict]]:
        """Run one bounded LLM extraction and atomically validate its ops.

        Durable/semantic decisions belong to the chat updater prompt.  This
        method only selects a token-bounded context, parses the response, and
        enforces operation shape, ids, sections, provenance, and atomicity.
        """
        users = sum(turn.role == "user" for turn in evicted_turns)
        assistants = sum(turn.role == "assistant" for turn in evicted_turns)
        self.evicted_user_assistant_pairs += min(users, assistants)
        if not evicted_turns:
            self.decision_reasons["skip:empty_batch"] += 1
            return [], []

        decision_reason = "call:llm_chat_update"
        self.decision_reasons[decision_reason] += 1
        # ``prepare_update`` has already partitioned all turns into complete,
        # oldest-first batches. Never apply a second suffix selection here:
        # doing so would silently omit turns that the adapter is about to
        # remove from its history.
        prompt_turns = list(evicted_turns)
        selection = UpdateMemorySelector(
            memory,
            self.token_estimator,
            max_candidate_entries=self.max_candidate_entries,
        ).select_for_update(prompt_turns, self.update_memory_token_budget)
        visible_entries = [memory.entries[entry.id] for entry in selection.entries]
        visible_ids = {entry.id for entry in visible_entries}
        system, messages = self._build_prompt(memory, prompt_turns, visible_entries)
        base_prompt_text = system + "\n" + "\n".join(
            str(message.get("content", "")) for message in messages
        )
        schema_system, schema_messages = self._build_prompt(memory, [], [])
        schema_prompt_text = schema_system + "\n" + "\n".join(
            str(message.get("content", "")) for message in schema_messages
        )
        memory_system, memory_messages = self._build_prompt(memory, [], visible_entries)
        memory_prompt_text = memory_system + "\n" + "\n".join(
            str(message.get("content", "")) for message in memory_messages
        )
        schema_tokens = self.token_estimator(schema_prompt_text)
        visible_component_tokens = max(
            0, self.token_estimator(memory_prompt_text) - schema_tokens
        )
        turn_component_tokens = max(
            0,
            self.token_estimator(base_prompt_text)
            - self.token_estimator(memory_prompt_text),
        )

        last_rejected: list[dict] = []
        for attempt in range(self.max_retries + 1):
            provider_before = self._provider_usage()
            try:
                response = self.llm.complete(system, messages, model=self.model)
            except Exception as exc:
                rejection = [{"op": None, "reason": f"LLM transport error: {exc}"}]
                self._record_call_report(
                    prompt_messages=messages,
                    base_prompt_text=base_prompt_text,
                    schema_tokens=schema_tokens,
                    visible_component_tokens=visible_component_tokens,
                    turn_component_tokens=turn_component_tokens,
                    prompt_turns=prompt_turns,
                    visible_entries=visible_entries,
                    response="",
                    provider_before=provider_before,
                    llm_ops_count=0,
                    rejected_count=1,
                    decision_reason=decision_reason,
                    required_overflow_tokens=selection.required_overflow_tokens,
                )
                if attempt < self.max_retries:
                    messages = self._retry_failure_messages(messages, rejection)
                    continue
                raise UpdateFailed(f"LLM transport error after retry exhaustion: {exc}") from exc

            ops = parse_memory_ops(response)
            if ops is None:
                rejection = [{"op": response, "reason": "response was not a JSON ops array"}]
                self._record_call_report(
                    prompt_messages=messages,
                    base_prompt_text=base_prompt_text,
                    schema_tokens=schema_tokens,
                    visible_component_tokens=visible_component_tokens,
                    turn_component_tokens=turn_component_tokens,
                    prompt_turns=prompt_turns,
                    visible_entries=visible_entries,
                    response=response,
                    provider_before=provider_before,
                    llm_ops_count=0,
                    rejected_count=1,
                    decision_reason=decision_reason,
                    required_overflow_tokens=selection.required_overflow_tokens,
                )
                if attempt < self.max_retries:
                    messages = self._retry_failure_messages(messages, rejection, response)
                    continue
                raise UpdateFailed(
                    f"Could not parse a JSON ops array after retry exhaustion: {response!r}"
                )

            hidden_id_rejections = [
                {"op": op, "reason": "UPDATE/SUPERSEDE id was not visible to updater"}
                for op in ops
                if isinstance(op, dict)
                and op.get("op") in {"UPDATE", "SUPERSEDE"}
                and isinstance(op.get("id"), str)
                and op.get("id") in memory.entries
                and op.get("id") not in visible_ids
            ]
            prompt_text = system + "\n" + "\n".join(
                str(message.get("content", "")) for message in messages
            )
            provider_after = self._provider_usage()
            provider_input = provider_output = None
            if (
                provider_before is not None
                and provider_after is not None
                and provider_after[0] > provider_before[0]
            ):
                provider_input = provider_after[1] - provider_before[1]
                provider_output = provider_after[2] - provider_before[2]
            call_report = UpdateTokenReport(
                call_index=len(self.token_reports) + 1,
                evicted_turn_count=len(prompt_turns),
                visible_memory_entry_count=len(visible_entries),
                visible_entries_by_section=dict(Counter(entry.section for entry in visible_entries)),
                estimator_policy="characters_divided_by_four",
                calls=1,
                system_tokens=schema_tokens,
                visible_memory_tokens=visible_component_tokens,
                evicted_turn_tokens=turn_component_tokens,
                output_tokens=self.token_estimator(response) if response else 0,
                retry_tokens=max(
                    0,
                    self.token_estimator(prompt_text)
                    - self.token_estimator(base_prompt_text),
                ),
                provider_input_tokens=provider_input,
                provider_output_tokens=provider_output,
                llm_ops_count=len(ops),
                rejected_ops_count=len(hidden_id_rejections),
                llm_call_required_reason=decision_reason,
                required_overflow_tokens=selection.required_overflow_tokens,
                mandatory_overflow_tokens=self.turn_selection_reports[-1]["mandatory_overflow_tokens"],
            )
            self.token_reports.append(call_report)
            if hidden_id_rejections:
                last_rejected = hidden_id_rejections
                if attempt < self.max_retries:
                    messages = self._retry_messages(messages, ops, hidden_id_rejections)
                    continue
                return [], hidden_id_rejections

            ops, structural_rejections = self._validate_structural_ops(ops, memory)
            ops = self._cap_ops(ops)
            ops = [
                op for op in ops
                if not (isinstance(op, dict) and op.get("op") == "NOOP")
            ]
            provenance_rejections = self._validate_provenance(ops, evicted_turns)
            rejected = [*structural_rejections, *provenance_rejections]
            if rejected:
                call_report.rejected_ops_count = len(rejected)
                last_rejected = rejected
                if attempt < self.max_retries:
                    messages = self._retry_messages(messages, ops, rejected)
                    continue
                return [], rejected
            if not ops:
                return [], []

            applied, rejected = memory.apply_ops_atomically(ops)
            call_report.rejected_ops_count = len(rejected)
            if not rejected:
                return applied, []
            last_rejected = rejected
            if attempt < self.max_retries:
                messages = self._retry_messages(messages, ops, rejected)

        return [], last_rejected

    @staticmethod
    def _retry_messages(messages: list[dict], ops: list[dict], rejected: list[dict]) -> list[dict]:
        feedback = {
            "rejected_ops": rejected,
            "instructions": [
                "Return a corrected full JSON array for the same turns.",
                "Do not repeat rejected UPDATE or SUPERSEDE ids.",
                "UPDATE/SUPERSEDE ids must be exact current memory entry ids like F1, U2, or G3.",
                "If no exact entry id exists, use ADD for new durable information or NOOP.",
                "Keep the corrected batch concise and avoid near-duplicate entries.",
            ],
        }
        return [
            *messages,
            {"role": "assistant", "content": json.dumps(ops, ensure_ascii=False)},
            {
                "role": "user",
                "content": (
                    "The previous memory ops were rejected by validation:\n"
                    f"{json.dumps(feedback, ensure_ascii=False, indent=2)}"
                ),
            },
        ]

    @staticmethod
    def _retry_failure_messages(
        messages: list[dict], rejected: list[dict], response: str | None = None
    ) -> list[dict]:
        retry = list(messages)
        if response:
            retry.append({"role": "assistant", "content": response})
        retry.append({
            "role": "user",
            "content": (
                "The previous updater attempt failed before producing valid operations. "
                "Retry the same turns and return only one valid JSON array of memory operations.\n"
                f"{json.dumps({'errors': rejected}, ensure_ascii=False)}"
            ),
        })
        return retry

    def _record_call_report(
        self,
        *,
        prompt_messages: list[dict],
        base_prompt_text: str,
        schema_tokens: int,
        visible_component_tokens: int,
        turn_component_tokens: int,
        prompt_turns: list[Turn],
        visible_entries: list,
        response: str,
        provider_before: tuple[int, int, int] | None,
        llm_ops_count: int,
        rejected_count: int,
        decision_reason: str,
        required_overflow_tokens: int,
    ) -> None:
        provider_after = self._provider_usage()
        provider_input = provider_output = None
        if (
            provider_before is not None
            and provider_after is not None
            and provider_after[0] > provider_before[0]
        ):
            provider_input = provider_after[1] - provider_before[1]
            provider_output = provider_after[2] - provider_before[2]
        prompt_text = "\n".join(str(message.get("content", "")) for message in prompt_messages)
        self.token_reports.append(UpdateTokenReport(
            call_index=len(self.token_reports) + 1,
            evicted_turn_count=len(prompt_turns),
            visible_memory_entry_count=len(visible_entries),
            visible_entries_by_section=dict(Counter(entry.section for entry in visible_entries)),
            estimator_policy="characters_divided_by_four",
            calls=1,
            system_tokens=schema_tokens,
            visible_memory_tokens=visible_component_tokens,
            evicted_turn_tokens=turn_component_tokens,
            output_tokens=self.token_estimator(response) if response else 0,
            retry_tokens=max(0, self.token_estimator(prompt_text) - self.token_estimator(base_prompt_text)),
            provider_input_tokens=provider_input,
            provider_output_tokens=provider_output,
            llm_ops_count=llm_ops_count,
            rejected_ops_count=rejected_count,
            llm_call_required_reason=decision_reason,
            required_overflow_tokens=required_overflow_tokens,
            mandatory_overflow_tokens=self.turn_selection_reports[-1]["mandatory_overflow_tokens"],
        ))

    def _validate_structural_ops(
        self,
        ops: list[dict],
        memory: Memory,
    ) -> tuple[list[dict], list[dict]]:
        """Validate operation shape without judging semantic content.

        The model owns retention choices. This gate only protects the store
        from malformed JSON, unknown sections, oversized text, and impossible
        operation fields before provenance and atomic application checks.
        """
        allowed_sections = {section.key for section in self.sections}
        rejected: list[dict] = []
        accepted: list[dict] = []
        for op in ops:
            reason: str | None = None
            if not isinstance(op, dict):
                reason = "operation must be an object"
            else:
                kind = op.get("op")
                if kind == "NOOP":
                    accepted.append(op)
                    continue
                elif kind == "ADD":
                    section = op.get("section")
                    text = op.get("text")
                    if section not in allowed_sections:
                        reason = f"unknown or disallowed section: {section}"
                    elif not isinstance(text, str) or not text.strip():
                        reason = "ADD text must be a non-empty string"
                    elif len(text) > 500:
                        reason = "ADD text exceeds 500 characters"
                    elif not isinstance(op.get("provenance"), list):
                        reason = "ADD provenance must be a list"
                elif kind == "UPDATE":
                    text = op.get("text")
                    if not isinstance(op.get("id"), str):
                        reason = "UPDATE id must be a string"
                    elif op.get("id") not in memory.entries:
                        reason = f"unknown UPDATE id: {op.get('id')}"
                    elif not isinstance(text, str) or not text.strip():
                        reason = "UPDATE text must be a non-empty string"
                    elif len(text) > 500:
                        reason = "UPDATE text exceeds 500 characters"
                    elif not isinstance(op.get("provenance"), list):
                        reason = "UPDATE provenance must be a list"
                elif kind == "SUPERSEDE":
                    if not isinstance(op.get("id"), str):
                        reason = "SUPERSEDE id must be a string"
                    elif op.get("id") not in memory.entries:
                        reason = f"unknown SUPERSEDE id: {op.get('id')}"
                else:
                    reason = f"unknown operation: {kind}"
            if reason is None:
                accepted.append(op)
            else:
                rejected.append({"op": op, "reason": reason})
        return accepted, rejected

    def _cap_ops(self, ops: list[dict]) -> list[dict]:
        """Apply only the configured operation-count safety cap."""
        limit = self.policy.max_ops_per_batch
        if limit is None or limit < 1:
            return ops
        return ops[:limit]

    @staticmethod
    def _validate_provenance(ops: list[dict], evicted_turns: list[Turn]) -> list[dict]:
        allowed_turn_ids = {turn.id for turn in evicted_turns}
        rejected: list[dict] = []

        for op in ops:
            if not isinstance(op, dict):
                continue

            if op.get("op") not in {"ADD", "UPDATE"}:
                continue

            provenance = op.get("provenance")
            if not isinstance(provenance, list) or not provenance:
                rejected.append({"op": op, "reason": "provenance must be a non-empty list"})
                continue

            invalid_ids = [
                turn_id
                for turn_id in provenance
                if not isinstance(turn_id, int) or turn_id not in allowed_turn_ids
            ]
            if invalid_ids:
                rejected.append(
                    {
                        "op": op,
                        "reason": f"provenance contains turn ids outside this batch: {invalid_ids}",
                    }
                )

        return rejected

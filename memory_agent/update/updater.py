"""Update pipeline that turns evicted turns into memory operations."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Callable

from memory_agent.clients.llm import LLMClient
from memory_agent.core.models import MemoryValue, SubjectIdentity, SubjectNormalizer
from memory_agent.core.sections import SectionConfig
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.normalization.chat import ChatSubjectNormalizer
from memory_agent.policies.structured import (
    AGENT_POLICY,
    StructuredMemoryPolicy,
    is_chat_policy,
    validate_policy_sections,
)
from memory_agent.update.heuristics import (
    ASSISTANT_ATTRIBUTED_RE,
    DURABLE_USER_STATE_RE,
    EXACT_VALUE_DATE_PATTERNS,
    EXACT_VALUE_PATTERNS,
    EXPLICIT_PROJECT_DENIAL_RE,
    GENERIC_NON_DURABLE_MEMORY_RE,
    ORDINARY_QUESTION_RE,
    PROGRESS_VALUE_RE,
    PERSONAL_SUBJECT_VALUE_PATTERNS,
    PROJECT_IMPLEMENTATION_STATE_RE,
    STATUS_CHANGE_CUE_RE,
    STATUS_VALUE_RE,
    STABLE_INSTRUCTION_RE,
    SUBJECT_VALUE_PATTERNS,
    SUBJECT_COUNT_PATTERNS,
    SUBJECT_VALUE_SECTION_RE,
    TECHNICAL_CONTEXT_RE,
    WHITESPACE_RE,
    content_words,
    status_change_cue_re,
    trim_conversational_frame,
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

_TURN_SUFFIX_RE = re.compile(r"\s*\(turns?\s+([0-9,\-\s]+)\)\s*$", re.IGNORECASE)
_ACCEPTANCE_RE = re.compile(
    r"\b(?:yes|agreed|sounds good|let'?s do (?:it|that)|go with that|accepted?)\b"
    r"|(?:同意|就這樣|採用這個|照這個做|可以，就這個)",
    re.IGNORECASE,
)
_REJECTION_RE = re.compile(
    r"\b(?:no|reject(?:ed)?|don'?t do that|do not do that|not that option|decline[ds]?)\b"
    r"|(?:拒絕|不要這個|不採用|換一個方案)",
    re.IGNORECASE,
)
_PURE_PROPOSAL_RESOLUTION_RE = re.compile(
    r"^\s*(?:(?:yes|agreed|accepted?|sounds good)(?:[,;:]?\s*(?:go with that|"
    r"let'?s do (?:it|that)))?|go with that|let'?s do (?:it|that)|"
    r"no(?:[,;:]?\s*(?:reject that proposal|don'?t do that))?|reject(?:ed)?(?: that proposal)?|"
    r"don'?t do that|do not do that|not that option|decline[ds]?|"
    r"(?:同意|就這樣|採用這個|照這個做|可以，就這個)(?:[，,]?就這樣)?|"
    r"拒絕|不要這個|不採用|換一個方案)\s*[.!。！]?\s*$",
    re.IGNORECASE,
)
_PROPOSAL_RE = re.compile(
    r"\b(?:i\s+(?:propose|suggest|recommend)|we\s+should|"
    r"my recommendation is)\b|(?:我(?:建議|提議)|建議採用)",
    re.IGNORECASE,
)
_EXPLICIT_STATE_RE = re.compile(
    r"\b(?P<subject>(?:the|my|our)\s+[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,4}|it)\s+"
    r"(?:is|was|became|has been)\s+(?P<state>planned|active|blocked|resumed|"
    r"paused|cancelled|canceled|complete|completed|shipped|failed|done|in progress)\b",
    re.IGNORECASE,
)
_ZH_EXPLICIT_STATE_RE = re.compile(
    r"(?P<subject>它|[\u4e00-\u9fffA-Za-z0-9_-]{1,12})"
    r"(?:目前)?(?:是|已經?|變成)"
    r"(?P<state>規劃中|進行中|受阻|恢復|暫停|完成|取消|失敗|已上線)",
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
    deterministic_ops_count: int = 0
    llm_ops_count: int = 0
    rejected_ops_count: int = 0
    llm_call_required_reason: str = "call:possible_durable_assertion"
    required_exact_subject_overflow_tokens: int = 0
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
class PreparedUpdate:
    trial_memory: Memory
    applied_ops: list[dict]
    rejected_ops: list[dict]
    base_revision: int
    _committed: bool = False

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
        subject_normalizer: SubjectNormalizer | None = None,
        identity_confidence_threshold: float = 0.85,
        max_candidate_entries: int = 8,
        max_legacy_candidate_entries: int = 4,
        enable_llm_gate: bool = False,
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
        # Direct construction historically behaved like the richer agent
        # updater. Product builders pass the practical policy explicitly.
        self.policy = policy or AGENT_POLICY
        self.subject_normalizer = subject_normalizer or ChatSubjectNormalizer()
        self.identity_confidence_threshold = identity_confidence_threshold
        self.max_candidate_entries = max_candidate_entries
        self.max_legacy_candidate_entries = max_legacy_candidate_entries
        self.enable_llm_gate = enable_llm_gate
        self.decision_reasons: Counter[str] = Counter()
        self.write_suppression_reasons: Counter[str] = Counter()
        self.lifecycle_diagnostics: Counter[str] = Counter()
        self.evicted_user_assistant_pairs = 0
        self.turn_selection_reports: list[dict] = []
        # Fail fast on profile/section mismatches instead of silently running
        # with retention behavior the caller did not intend.
        validate_policy_sections(self.policy, sections)
        self._section_key_by_prefix = {section.prefix.lower(): section.key for section in sections}

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
            "calls_skipped_by_deterministic_gating": sum(v for k, v in self.decision_reasons.items() if k.startswith("skip:")),
            "decision_reasons": dict(self.decision_reasons),
            "write_suppression_reasons": dict(self.write_suppression_reasons),
            "suppressed_write_count": sum(self.write_suppression_reasons.values()),
            "lifecycle_diagnostics": dict(self.lifecycle_diagnostics),
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
        }

    # Sections whose entries are always shown to the updater regardless of
    # lexical overlap: they are few, and the dedup/supersede rules depend on
    # the LLM seeing them.
    _ALWAYS_CONTEXT_SECTIONS = frozenset({"preferences", "goal", "status_changes"})

    def _turns_within_budget(self, turns: list[Turn]) -> list[Turn]:
        groups = self._semantic_turn_groups(turns)
        prompt_turns_by_group = {
            id(group): self._prompt_turns_for_group(group) for group in groups
        }
        budget = self.evicted_turn_token_budget
        selected: list[TurnGroup] = []
        used = 0
        if budget is None:
            selected = groups
        else:
            # The newest group is mandatory: in particular this preserves a
            # latest unresolved user or its complete request/response exchange.
            for group in reversed(groups):
                tokens = sum(
                    self.token_estimator(turn.content)
                    for turn in prompt_turns_by_group[id(group)]
                )
                if group.mandatory or used + tokens <= budget:
                    selected.append(group)
                    used += tokens
                else:
                    # Once a group is dropped, older groups are not backfilled:
                    # updater context remains a contiguous suffix.
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
                {"type": group.group_type, "turn_ids": [turn.id for turn in group.turns],
                 "mandatory": group.mandatory}
                for group in groups
            ],
        })
        return [turn for turn in turns if turn.id in selected_ids]

    def _prompt_turns_for_group(self, group: TurnGroup) -> tuple[Turn, ...]:
        """Keep only source roles that can affect practical chat memory.

        Ordinary assistant answers are transient under chat retention policy and
        can be much larger than the user assertion they answer. Proposal
        resolution, correction, and tool-result groups retain their full source
        context because ownership or confirmation depends on multiple roles.
        """
        has_concrete_proposal = any(
            turn.role == "assistant" and _PROPOSAL_RE.search(turn.content)
            for turn in group.turns
        )
        has_progress_source = self._has_section("progress") and self._is_substantive_exchange(
            group.turns
        )
        if (
            is_chat_policy(self.policy)
            and group.group_type == "user_assistant"
            and not has_concrete_proposal
            and not has_progress_source
        ):
            return tuple(turn for turn in group.turns if turn.role == "user")
        return group.turns

    @staticmethod
    def _is_substantive_exchange(turns: tuple[Turn, ...] | list[Turn]) -> bool:
        """Conservative gate for exchanges worth a compactable progress rollup."""
        has_user = any(turn.role == "user" and turn.content.strip() for turn in turns)
        assistant_text = " ".join(
            turn.content.strip()
            for turn in turns
            if turn.role == "assistant" and turn.content.strip()
        )
        return (
            has_user
            and len(assistant_text) >= 180
            and len(content_words(assistant_text)) >= 25
        )

    @staticmethod
    def _semantic_turn_groups(turns: list[Turn]) -> list[TurnGroup]:
        """Group contiguous conversational exchanges without slicing messages."""
        raw: list[tuple[list[Turn], str]] = []
        current: list[Turn] = []
        for turn in turns:
            if turn.role == "user" and current:
                raw.append((current, MemoryUpdater._group_type(current)))
                current = []
            current.append(turn)
        if current:
            raw.append((current, MemoryUpdater._group_type(current)))

        context_re = re.compile(
            r"(?:\b(?:yes|agreed|accept|sounds good|go with|no|reject|instead|actually|correction|changed my mind|no longer|not anymore)\b|同意|接受|拒絕|不要|改成|更正|其實|不再)",
            re.I,
        )
        merged: list[tuple[list[Turn], str]] = []
        for group, kind in raw:
            user_text = next((t.content for t in group if t.role == "user"), "")
            if merged and context_re.search(user_text):
                prior, _prior_kind = merged.pop()
                kind = "correction_context" if re.search(r"(?:\b(?:instead|actually|correction|changed my mind|no longer|not anymore)\b|改成|更正|其實|不再)", user_text, re.I) else "acceptance_context"
                group = [*prior, *group]
            merged.append((group, kind))
        return [
            TurnGroup(tuple(group), kind, mandatory=index == len(merged) - 1)
            for index, (group, kind) in enumerate(merged)
        ]

    @staticmethod
    def _group_type(turns: list[Turn]) -> str:
        roles = {turn.role for turn in turns}
        if "tool" in roles or any("[tool_call]" in turn.content for turn in turns):
            return "tool_call_result"
        if turns and turns[0].role == "user" and len(turns) == 1:
            return "unresolved_user"
        if turns and turns[0].role == "user" and "assistant" in roles:
            return "user_assistant"
        return "standalone"

    def _select_update_context_entries(self, memory: Memory, evicted_turns: list[Turn]) -> list:
        """Pick the memory entries most relevant to the evicted turns.

        UPDATE and SUPERSEDE require the LLM to cite an exact entry id. When
        the whole memory (a hundred-plus entries) is dumped into the prompt, a
        small updater model reliably fails to spot the one conflicting entry,
        so stale values survive forever. Selecting a focused candidate set by
        lexical overlap makes conflict detection tractable. Superseded entries
        that overlap are kept too, so old invalid facts do not get re-added.
        """
        query_words = content_words(
            "\n".join(turn.content for turn in evicted_turns if turn.role in {"user", "assistant"})
        )

        always = []
        scored = []
        for entry in memory.entries.values():
            if entry.section in self._ALWAYS_CONTEXT_SECTIONS:
                always.append(entry)
                continue
            overlap = len(query_words & content_words(entry.text))
            if overlap <= 0:
                continue
            score = overlap * 3.0
            if entry.status == "active":
                score += 2.0
            if entry.provenance:
                score += min(max(entry.provenance), 1000) / 1000.0
            scored.append((score, entry))

        scored.sort(key=lambda item: (-item[0], item[1].id))
        budget = max(0, self.update_context_max_entries - len(always))
        return [*always, *[entry for _score, entry in scored[:budget]]]

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

    def _llm_decision(
        self,
        memory: Memory,
        evicted_turns: list[Turn],
        deterministic_ops: list[dict],
    ) -> str:
        """Conservatively decide whether deterministic extraction is complete."""
        user_turns = [turn for turn in evicted_turns if turn.role == "user" and turn.content.strip()]
        if not user_turns:
            return "skip:no_durable_assertion"
        combined = " ".join(turn.content for turn in user_turns)
        has_deterministic_resolution = any(
            isinstance(op, dict)
            and op.get("op") == "ADD"
            and op.get("section") == "decisions"
            and str(op.get("text", "")).startswith(
                ("Accepted strategy:", "Rejected proposal:")
            )
            for op in deterministic_ops
        )
        if has_deterministic_resolution and all(
            _PURE_PROPOSAL_RESOLUTION_RE.fullmatch(turn.content)
            for turn in user_turns
        ):
            return "skip:deterministic_ops_fully_cover_batch"
        if _ACCEPTANCE_RE.search(combined):
            return "call:user_acceptance_ambiguous"
        if re.search(r"\b(?:correction|changed my mind|no longer|not anymore|instead|actually|contradict(?:s|ion|ory)?)\b", combined, re.I):
            return "call:unresolved_subject_conflict"
        if self._is_ordinary_non_durable_batch(
            evicted_turns, cue_re=status_change_cue_re(self.policy)
        ):
            if self._has_section("progress") and self._is_substantive_exchange(
                evicted_turns
            ):
                return "call:progress_rollup_source"
            return "skip:no_durable_assertion"
        covered_turn_ids = {
            turn_id
            for op in deterministic_ops
            if isinstance(op, dict)
            for turn_id in op.get("provenance", [])
            if isinstance(turn_id, int)
        }
        durable_turn_ids = {
            turn.id
            for turn in user_turns
            if DURABLE_USER_STATE_RE.search(turn.content)
            or status_change_cue_re(self.policy).search(turn.content)
        }
        if durable_turn_ids and durable_turn_ids <= covered_turn_ids:
            return "skip:deterministic_ops_fully_cover_batch"
        return "call:possible_durable_assertion"

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
        return result.applied_ops, result.rejected_ops

    def prepare_update(self, memory: Memory, evicted_turns: list[Turn]) -> PreparedUpdate:
        trial, base_revision = memory.transaction_snapshot()
        applied, rejected = self._update_trial(trial, evicted_turns)
        return PreparedUpdate(trial, applied, rejected, base_revision)

    def _update_trial(self, memory: Memory, evicted_turns: list[Turn]) -> tuple[list[dict], list[dict]]:
        users = sum(turn.role == "user" for turn in evicted_turns)
        assistants = sum(turn.role == "assistant" for turn in evicted_turns)
        self.evicted_user_assistant_pairs += min(users, assistants)
        deterministic_ops = self._deterministic_ops(memory, evicted_turns)
        _debug_ops("deterministic before filter", deterministic_ops)
        deterministic_ops = self._apply_policy_filter(
            deterministic_ops,
            memory,
            evicted_turns,
            apply_cap=False,
        )
        decision_reason = self._llm_decision(memory, evicted_turns, deterministic_ops)
        self.decision_reasons[decision_reason] += 1
        _debug_ops("deterministic after filter", deterministic_ops)
        det_applied: list[dict] = []
        if deterministic_ops:
            det_applied, _det_rejected = memory.apply_ops_atomically(deterministic_ops)

        should_skip = (
            decision_reason == "skip:no_durable_assertion" and is_chat_policy(self.policy)
        ) or (self.enable_llm_gate and decision_reason.startswith("skip:"))
        if should_skip:
            consolidated = [
                *self._consolidate_lifecycle(memory),
                *self._consolidate_near_duplicates(memory),
            ]
            return det_applied + consolidated, []

        prompt_turns = self._turns_within_budget(evicted_turns)
        selection = UpdateMemorySelector(
            memory, self.token_estimator, self.subject_normalizer,
            self.identity_confidence_threshold,
            max_legacy_fallback_entries=self.max_legacy_candidate_entries,
            max_candidate_entries=self.max_candidate_entries,
        ).select_for_update(
            prompt_turns, self.update_memory_token_budget
        )
        # Migration-on-touch is an explicit atomic operation over selected legacy
        # entries only. Normalization never mutates live entries during discovery.
        migration_ops = []
        for entry in selection.entries:
            if entry.subject_identity is not None and entry.value is not None:
                continue
            normalized = self.subject_normalizer.normalize(entry.text)
            if normalized is None or normalized[0].confidence < self.identity_confidence_threshold:
                continue
            migration_ops.append({
                "op": "UPDATE", "id": entry.id, "text": entry.text,
                "provenance": list(entry.provenance),
                "subject_identity": normalized[0], "value": normalized[1],
            })
        migrated_applied: list[dict] = []
        if migration_ops:
            migrated_applied, migration_rejected = memory.apply_ops_atomically(migration_ops)
            if migration_rejected:
                migrated_applied = []
        visible_entries = [memory.entries[entry.id] for entry in selection.entries]
        visible_ids = {entry.id for entry in visible_entries}
        system, messages = self._build_prompt(memory, prompt_turns, visible_entries)
        base_prompt_text = system + "\n" + "\n".join(str(m.get("content", "")) for m in messages)
        schema_system, schema_messages = self._build_prompt(memory, [], [])
        schema_prompt_text = schema_system + "\n" + "\n".join(
            str(m.get("content", "")) for m in schema_messages
        )
        memory_system, memory_messages = self._build_prompt(memory, [], visible_entries)
        memory_prompt_text = memory_system + "\n" + "\n".join(
            str(m.get("content", "")) for m in memory_messages
        )
        schema_tokens = self.token_estimator(schema_prompt_text)
        visible_component_tokens = max(
            0, self.token_estimator(memory_prompt_text) - schema_tokens
        )
        turn_component_tokens = max(
            0, self.token_estimator(base_prompt_text) - self.token_estimator(memory_prompt_text)
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
                    deterministic_ops=deterministic_ops,
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
                    deterministic_ops=deterministic_ops,
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
            ops = self._normalize_ops(ops, memory)
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
            if provider_before is not None and provider_after is not None and provider_after[0] > provider_before[0]:
                provider_input = provider_after[1] - provider_before[1]
                provider_output = provider_after[2] - provider_before[2]
            retry_tokens = max(0, self.token_estimator(prompt_text) - self.token_estimator(base_prompt_text))
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
                retry_tokens=retry_tokens,
                provider_input_tokens=provider_input,
                provider_output_tokens=provider_output,
                deterministic_ops_count=len(deterministic_ops),
                llm_ops_count=len(ops),
                rejected_ops_count=len(hidden_id_rejections),
                llm_call_required_reason=decision_reason,
                required_exact_subject_overflow_tokens=selection.required_overflow_tokens,
                mandatory_overflow_tokens=self.turn_selection_reports[-1]["mandatory_overflow_tokens"],
            )
            self.token_reports.append(call_report)
            if hidden_id_rejections:
                applied, rejected = [], hidden_id_rejections
                last_rejected = rejected
                if attempt < self.max_retries:
                    messages = self._retry_messages(messages, ops, rejected)
                    continue
                return det_applied, rejected
            ops = self._drop_duplicate_deterministic_adds(ops, memory)
            _debug_ops("llm before filter", ops)
            ops = self._apply_policy_filter(ops, memory, evicted_turns)
            ops = self._suppress_redundant_and_transient_adds(
                ops, memory, evicted_turns
            )
            _debug_ops("llm after filter", ops)
            ops = [
                op for op in ops if not (isinstance(op, dict) and op.get("op") == "NOOP")
            ]
            if not ops:
                consolidated = [
                    *self._consolidate_lifecycle(memory),
                    *self._consolidate_near_duplicates(memory),
                ]
                return det_applied + migrated_applied + consolidated, []

            provenance_rejections = self._validate_provenance(ops, evicted_turns)
            if provenance_rejections:
                applied, rejected = [], provenance_rejections
            else:
                applied, rejected = memory.apply_ops_atomically(ops)
            call_report.rejected_ops_count = len(rejected)

            if not rejected:
                consolidated = [
                    *self._consolidate_lifecycle(memory),
                    *self._consolidate_near_duplicates(memory),
                ]
                return det_applied + migrated_applied + applied + consolidated, []

            last_rejected = rejected
            if attempt < self.max_retries:
                messages = self._retry_messages(messages, ops, rejected)

        return det_applied, last_rejected

    @staticmethod
    def _consolidate_near_duplicates(memory: Memory) -> list[dict]:
        """Locally supersede high-overlap entries without another LLM call."""
        eligible = {"facts", "goal", "open_questions", "preferences", "status_changes"}
        active = [
            entry
            for entry in memory.entries.values()
            if entry.status == "active" and entry.section in eligible
        ]

        def duplicate_words(text: str) -> set[str]:
            return {word.strip(".,;:!?()[]{}") for word in content_words(text)}

        ops: list[dict] = []
        remaining = list(active)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            seed_words = duplicate_words(seed.text)
            unmatched = []
            for candidate in remaining:
                if candidate.section != seed.section:
                    unmatched.append(candidate)
                    continue
                candidate_words = duplicate_words(candidate.text)
                union = seed_words | candidate_words
                overlap = len(seed_words & candidate_words) / len(union) if union else 0.0
                seed_key = MemoryUpdater._text_key(seed.text)
                candidate_key = MemoryUpdater._text_key(candidate.text)
                duplicate = (
                    min(len(seed_words), len(candidate_words)) >= 5
                    and (
                        overlap >= 0.60
                        or seed_key in candidate_key
                        or candidate_key in seed_key
                    )
                )
                (cluster if duplicate else unmatched).append(candidate)
            remaining = unmatched
            if len(cluster) < 2:
                continue
            keep = max(cluster, key=lambda entry: (max(entry.provenance or [0]), entry.id))
            provenance = sorted({turn_id for entry in cluster for turn_id in entry.provenance})
            ops.append({
                "op": "UPDATE",
                "id": keep.id,
                "text": keep.text,
                "provenance": provenance,
            })
            ops.extend(
                {
                    "op": "SUPERSEDE",
                    "id": entry.id,
                    "reason": f"Near-duplicate consolidated into {keep.id}.",
                }
                for entry in cluster
                if entry.id != keep.id
            )
        if not ops:
            return []
        applied, rejected = memory.apply_ops_atomically(ops)
        return applied if not rejected else []

    @staticmethod
    def _consolidate_latest_subject_values(
        memory: Memory,
        normalizer: SubjectNormalizer | None = None,
        confidence_threshold: float = 0.85,
    ) -> list[dict]:
        """Keep a bounded value history for confidently identical subjects."""
        normalizer = normalizer or ChatSubjectNormalizer()
        ops: list[dict] = []
        groups: dict[tuple[str, str, str, str | None, str | None], list] = {}
        for entry in memory.entries.values():
            if entry.status != "active" or entry.section not in {
                "facts", "goal", "status_changes", "preferences"
            }:
                continue
            if entry.subject_identity is None or entry.value is None:
                normalized = normalizer.normalize(entry.text)
                if normalized is None or normalized[0].confidence < confidence_threshold:
                    continue
                # Legacy entries are migrated only by the update selector's
                # explicit migration-on-touch operation.
                continue
            identity, value = entry.subject_identity, entry.value
            if identity.confidence < confidence_threshold:
                continue
            if not MemoryUpdater._identity_is_specific(identity):
                continue
            key = (identity.namespace, identity.entity, identity.attribute, identity.qualifier, value.unit)
            groups.setdefault(key, []).append(entry)
        for matches in groups.values():
            if len(matches) < 2:
                continue
            ordered = sorted(
                matches,
                key=lambda entry: (max(entry.provenance or [0]), entry.id),
            )
            keep = ordered[-1]
            provenance = sorted({turn_id for entry in matches for turn_id in entry.provenance})
            values: list[str] = []
            for entry in ordered:
                for rendered in MemoryUpdater._entry_value_history(entry):
                    if not values or rendered != values[-1]:
                        values.append(rendered)
            text = re.sub(
                r"\s+Value history \(earliest→latest\):.*?\.\s*$",
                "",
                keep.text,
            )
            if len(values) > 1:
                history = " → ".join(values[-4:])
                text = f"{text} Value history (earliest→latest): {history}."
            ops.append({
                "op": "UPDATE",
                "id": keep.id,
                "text": text,
                "provenance": provenance,
                "subject_identity": keep.subject_identity,
                "value": keep.value,
            })
            for entry in matches:
                if entry.id == keep.id:
                    continue
                ops.append({
                    "op": "SUPERSEDE",
                    "id": entry.id,
                    "reason": f"Older subject value superseded by {keep.id}.",
                })
        if not ops:
            return []
        applied, rejected = memory.apply_ops_atomically(ops)
        return applied if not rejected else []

    def _consolidate_lifecycle(self, memory: Memory) -> list[dict]:
        ops = self._consolidate_latest_subject_values(
            memory,
            self.subject_normalizer,
            self.identity_confidence_threshold,
        )
        if not ops:
            self.lifecycle_diagnostics["uncertain_groups_skipped"] += 1
            return []
        superseded = sum(
            isinstance(op, dict) and op.get("op") == "SUPERSEDE" for op in ops
        )
        histories = sum(
            isinstance(op, dict)
            and op.get("op") == "UPDATE"
            and "Value history (earliest→latest):" in str(op.get("text", ""))
            for op in ops
        )
        self.lifecycle_diagnostics["lifecycle_groups"] += sum(
            isinstance(op, dict) and op.get("op") == "UPDATE" for op in ops
        )
        self.lifecycle_diagnostics["timeline_entries_created"] += histories
        self.lifecycle_diagnostics["redundant_entries_superseded"] += superseded
        self.lifecycle_diagnostics["important_transitions_preserved"] += histories
        return ops

    @staticmethod
    def _identity_is_specific(identity) -> bool:
        entity = identity.entity.strip().lower()
        attribute = identity.attribute.strip().lower()
        if not entity or entity == attribute:
            return False
        generic = {
            "goal", "target", "budget", "rate", "duration", "value",
            "my goal", "our goal", "the goal", "a goal",
        }
        if entity in generic:
            return False
        distinguishing = content_words(entity) - {
            attribute, "goal", "target", "budget", "rate", "duration",
            "set", "reach", "reached", "trying", "want", "wants",
        }
        required = 2 if attribute in {"goal", "target"} else 1
        # ``content_words`` intentionally favors Latin words of length >= 3.
        # A high-confidence typed identity should remain usable when its specific
        # entity is written in another script.
        if required == 1 and not distinguishing:
            return any(ord(char) > 127 and char.isalnum() for char in entity)
        return len(distinguishing) >= required

    @staticmethod
    def _render_memory_value(value) -> str:
        unit = value.unit or ""
        if unit in {"$", "€", "£"}:
            return f"{unit}{value.value}"
        return f"{value.value} {unit}".strip()

    @staticmethod
    def _entry_value_history(entry) -> list[str]:
        match = re.search(
            r"Value history \(earliest→latest\):\s*(.+?)\.\s*$",
            entry.text,
        )
        if match:
            values = [part.strip() for part in match.group(1).split("→")]
            if values and all(values):
                return values
        return [MemoryUpdater._render_memory_value(entry.value)]

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
        deterministic_ops: list[dict],
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
            deterministic_ops_count=len(deterministic_ops),
            llm_ops_count=llm_ops_count,
            rejected_ops_count=rejected_count,
            llm_call_required_reason=decision_reason,
            required_exact_subject_overflow_tokens=required_overflow_tokens,
            mandatory_overflow_tokens=self.turn_selection_reports[-1]["mandatory_overflow_tokens"],
        ))

    def _normalize_ops(self, ops: list[dict], memory: Memory) -> list[dict]:
        normalized_ops: list[dict] = []
        for op in ops:
            if not isinstance(op, dict):
                normalized_ops.append(op)
                continue

            normalized = dict(op)
            kind = normalized.get("op")
            if kind == "ADD":
                section = normalized.get("section")
                if isinstance(section, str):
                    section_key = self._section_key_by_prefix.get(section.lower())
                    if section_key is not None:
                        normalized["section"] = section_key
                self._normalize_text_provenance(normalized)
            elif (
                isinstance(kind, str)
                and kind.lower() in self._section_key_by_prefix
                and "section" not in normalized
                and "text" in normalized
                and "provenance" in normalized
            ):
                normalized["op"] = "ADD"
                normalized["section"] = self._section_key_by_prefix[kind.lower()]
                self._normalize_text_provenance(normalized)
            elif kind in {"UPDATE", "SUPERSEDE"}:
                entry_id = self._normalize_entry_id(normalized.get("id"), memory)
                if entry_id is not None:
                    normalized["id"] = entry_id
                if kind == "UPDATE":
                    self._normalize_text_provenance(normalized)

            normalized_ops.append(normalized)
        return normalized_ops

    def _deterministic_ops(self, memory: Memory, evicted_turns: list[Turn]) -> list[dict]:
        ops: list[dict] = []
        ops.extend(self._deterministic_proposal_resolution_ops(memory, evicted_turns))
        explicit_state_ops = self._deterministic_explicit_state_ops(memory, evicted_turns)
        ops.extend(explicit_state_ops)
        ops.extend(self._deterministic_preference_ops(memory, evicted_turns))
        ops.extend(self._deterministic_project_state_ops(memory, evicted_turns))
        if self.policy.allow_exact_values:
            ops.extend(self._deterministic_exact_value_ops(memory, evicted_turns))
        if self.policy.allow_deterministic_subject_values:
            ops.extend(self._deterministic_subject_value_ops(memory, evicted_turns))
        covered_state_turns = {
            turn_id
            for op in explicit_state_ops
            for turn_id in op.get("provenance", [])
        }
        ops.extend(
            op
            for op in self._deterministic_status_change_ops(memory, evicted_turns)
            if not covered_state_turns.intersection(op.get("provenance", []))
        )
        return ops

    def _deterministic_proposal_resolution_ops(
        self, memory: Memory, evicted_turns: list[Turn]
    ) -> list[dict]:
        if not self._has_section("decisions"):
            return []
        seen = self._active_text_keys(memory, "decisions")
        generated: list[dict] = []
        for index, turn in enumerate(evicted_turns):
            if turn.role != "user":
                continue
            accepted = bool(_ACCEPTANCE_RE.search(turn.content))
            rejected = bool(_REJECTION_RE.search(turn.content)) and not accepted
            if not accepted and not rejected:
                continue
            previous = next(
                (
                    candidate
                    for candidate in reversed(evicted_turns[:index])
                    if candidate.role in {"assistant", "user"}
                ),
                None,
            )
            if previous is None or previous.role != "assistant":
                continue
            proposal = self._proposal_summary(previous.content)
            if proposal is None:
                continue
            prefix = "Accepted strategy" if accepted else "Rejected proposal"
            text = f"{prefix}: {proposal}"
            if self._has_seen_text(text, seen):
                continue
            seen.add(self._text_key(text))
            generated.append({
                "op": "ADD",
                "section": "decisions",
                "text": text,
                "provenance": [previous.id, turn.id],
            })
        return generated

    @staticmethod
    def _proposal_summary(content: str) -> str | None:
        prose = MemoryUpdater._strip_code_fences(content).strip()
        if not prose or not _PROPOSAL_RE.search(prose):
            return None
        sentence = re.split(r"(?<=[.!?。！？])(?:\s+|$)|\n", prose, maxsplit=1)[0].strip()
        sentence = re.sub(
            r"^(?:i\s+(?:propose|suggest|recommend)(?:\s+that)?|"
            r"my recommendation is|we\s+(?:can|could|should)|let'?s)\s+",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip()
        sentence = re.sub(
            r"^(?:我(?:建議(?:採用)?|提議)|我們可以|建議採用)\s*",
            "",
            sentence,
        ).strip()
        if not sentence or len(sentence) > 300:
            return None
        return sentence.rstrip(".")

    def _deterministic_explicit_state_ops(
        self, memory: Memory, evicted_turns: list[Turn]
    ) -> list[dict]:
        if not self._has_section("status_changes"):
            return []
        seen = self._active_text_keys(memory, "status_changes")
        generated: list[dict] = []
        last_subject: str | None = None
        for turn in evicted_turns:
            if turn.role != "user":
                continue
            for match in _EXPLICIT_STATE_RE.finditer(turn.content):
                subject = match.group("subject").lower()
                if subject == "it":
                    if last_subject is None:
                        continue
                    subject = last_subject
                else:
                    subject = re.sub(r"^(?:the|my|our)\s+", "", subject)
                    last_subject = subject
                state = match.group("state").lower()
                state = {"completed": "complete", "done": "complete"}.get(state, state)
                text = f"State: {subject} is {state}."
                if self._has_seen_text(text, seen):
                    continue
                seen.add(self._text_key(text))
                generated.append({
                    "op": "ADD",
                    "section": "status_changes",
                    "text": text,
                    "provenance": [turn.id],
                    "subject_identity": SubjectIdentity(
                        "chat", subject, "state", confidence=0.95
                    ),
                    "value": MemoryValue(state),
                })
            for match in _ZH_EXPLICIT_STATE_RE.finditer(turn.content):
                subject = match.group("subject")
                if subject == "它":
                    if last_subject is None:
                        continue
                    subject = last_subject
                else:
                    last_subject = subject
                raw_state = match.group("state")
                state = {
                    "規劃中": "planned",
                    "進行中": "active",
                    "受阻": "blocked",
                    "恢復": "resumed",
                    "暫停": "paused",
                    "完成": "complete",
                    "取消": "cancelled",
                    "失敗": "failed",
                    "已上線": "shipped",
                }[raw_state]
                text = f"State: {subject} is {state}."
                if self._has_seen_text(text, seen):
                    continue
                seen.add(self._text_key(text))
                generated.append({
                    "op": "ADD",
                    "section": "status_changes",
                    "text": text,
                    "provenance": [turn.id],
                    "subject_identity": SubjectIdentity(
                        "chat", subject, "state", confidence=0.95
                    ),
                    "value": MemoryValue(state),
                })
        return generated

    def _deterministic_project_state_ops(
        self,
        memory: Memory,
        evicted_turns: list[Turn],
    ) -> list[dict]:
        """Retain explicit ongoing/completed implementation state as facts."""
        if not self._has_section("facts"):
            return []
        seen = self._active_text_keys(memory, "facts")
        generated: list[dict] = []
        # One newest explicit project event per eviction batch gives long-range
        # summaries broader coverage without turning every implementation
        # question into a permanent entry.
        for turn in reversed(evicted_turns):
            if turn.role != "user":
                continue
            prose = self._strip_code_fences(turn.content).split("->->", 1)[0]
            for sentence in re.split(r"(?<=[.!?])\s+|\n", prose):
                sentence = WHITESPACE_RE.sub(" ", sentence).strip()
                if not sentence or not PROJECT_IMPLEMENTATION_STATE_RE.search(sentence):
                    continue
                trimmed = trim_conversational_frame(sentence)
                if PROJECT_IMPLEMENTATION_STATE_RE.search(trimmed):
                    sentence = trimmed
                state = "Completed state" if re.search(r"\b(?:completed|implemented|fixed|finished|done)\b", sentence, re.I) else "Ongoing state"
                text = f"{state}: {sentence}"
                if self._has_seen_text(text, seen):
                    continue
                seen.add(self._text_key(text))
                generated.append({
                    "op": "ADD",
                    "section": "facts",
                    "text": text,
                    "provenance": [turn.id],
                })
                return generated
        return generated

    def _deterministic_preference_ops(
        self,
        memory: Memory,
        evicted_turns: list[Turn],
    ) -> list[dict]:
        """Guarantee stable user instructions survive noisy multi-fact batches."""
        if not self._has_section("preferences"):
            return []
        seen = self._active_text_keys(memory, "preferences")
        generated: list[dict] = []
        for turn in evicted_turns:
            if turn.role != "user" or not STABLE_INSTRUCTION_RE.search(turn.content):
                continue
            prose = self._strip_code_fences(turn.content).split("->->", 1)[0].strip()
            sentence = re.split(r"(?<=[.!?])\s+|\n", prose, maxsplit=1)[0].strip()
            if not sentence:
                continue
            text = f"Stable preference: {sentence}"
            if self._has_seen_text(text, seen):
                continue
            seen.add(self._text_key(text))
            generated.append({
                "op": "ADD",
                "section": "preferences",
                "text": text,
                "provenance": [turn.id],
            })
        return generated

    def _apply_policy_filter(
        self,
        ops: list[dict],
        memory: Memory,
        evicted_turns: list[Turn],
        *,
        apply_cap: bool = True,
    ) -> list[dict]:
        """Apply deterministic retention constraints after LLM extraction."""
        if not is_chat_policy(self.policy):
            return ops

        allowed_sections = {section.key for section in self.sections}
        disallowed_sections = {"timeline", "tool_facts", "exact_values"}
        ordinary_question = self._is_ordinary_non_durable_batch(
            evicted_turns, cue_re=status_change_cue_re(self.policy)
        )
        filtered: list[dict] = []

        for op in ops:
            if not isinstance(op, dict):
                filtered.append(op)
                continue

            kind = op.get("op")
            if kind == "ADD":
                section = op.get("section")
                text = op.get("text")
                if section not in allowed_sections or section in disallowed_sections:
                    continue
                if not isinstance(text, str) or GENERIC_NON_DURABLE_MEMORY_RE.search(text):
                    continue
                if ASSISTANT_ATTRIBUTED_RE.search(text):
                    continue
                if section == "progress":
                    provenance = set(op.get("provenance") or [])
                    turns_by_id = {turn.id: turn for turn in evicted_turns}
                    roles = {
                        turns_by_id[turn_id].role
                        for turn_id in provenance
                        if turn_id in turns_by_id
                    }
                    if not {"user", "assistant"} <= roles:
                        continue
                canonical = self._canonical_chat_entry_text(text, section)
                if canonical is None:
                    continue
                op["text"] = canonical
                explicit_denial = section == "status_changes" and bool(
                    EXPLICIT_PROJECT_DENIAL_RE.search(text)
                )
                subject_identity = op.get("subject_identity")
                typed_state = (
                    isinstance(subject_identity, SubjectIdentity)
                    and subject_identity.attribute == "state"
                    and isinstance(op.get("value"), MemoryValue)
                )
                if (
                    ordinary_question
                    and section != "progress"
                    and not explicit_denial
                    and not typed_state
                ):
                    continue
            elif kind == "UPDATE":
                text = op.get("text")
                if not isinstance(text, str) or GENERIC_NON_DURABLE_MEMORY_RE.search(text):
                    continue
                if ASSISTANT_ATTRIBUTED_RE.search(text):
                    continue
                entry_id = op.get("id")
                entry = memory.entries.get(entry_id) if isinstance(entry_id, str) else None
                canonical = self._canonical_chat_entry_text(text, entry.section if entry else None)
                if canonical is None:
                    continue
                op["text"] = canonical
                if entry is not None and entry.section in disallowed_sections:
                    continue
                if ordinary_question:
                    continue

            filtered.append(op)

        return self._cap_ops(filtered) if apply_cap else filtered

    @staticmethod
    def _canonical_chat_entry_text(text: str, section: str | None) -> str | None:
        text = WHITESPACE_RE.sub(" ", text).strip()
        prefix = ""
        body = text
        prefix_match = re.match(
            r"^(Ongoing state:|Completed state:|Goal:|Constraint:|"
            r"Stable preference:|User stated:)\s*",
            body,
        )
        if prefix_match:
            prefix, body = text[: prefix_match.end()], text[prefix_match.end() :]
        text = prefix + trim_conversational_frame(body)
        if not text or text.endswith(("…", "...", ":", ",", ";", "-")):
            return None
        # Quarantine oversized model prose instead of creating a sliced fragment.
        if len(text) > 500:
            return None
        raw = re.sub(r"^(?:the\s+)?user\s+(?:asked|requested|wants?)\s+(?:me\s+)?(?:to\s+)?", "", text, flags=re.I)
        if raw != text:
            text = raw.strip()
        if text.startswith(("Ongoing state:", "Completed state:", "Goal:", "Constraint:", "Stable preference:", "User stated:")):
            return text
        prefix = {
            "goal": "Goal",
            "preferences": "Stable preference",
            "status_changes": "Ongoing state",
        }.get(section)
        return f"{prefix}: {text}" if prefix else text

    def _cap_ops(self, ops: list[dict]) -> list[dict]:
        limit = self.policy.max_ops_per_batch
        if limit is None or limit < 1:
            return ops

        supersedes = [
            op for op in ops if isinstance(op, dict) and op.get("op") == "SUPERSEDE"
        ]
        replacements = [
            op for op in ops if isinstance(op, dict) and op.get("op") == "ADD"
        ]
        if supersedes and replacements:
            return [*supersedes, replacements[0]]

        actionable = [
            op
            for op in ops
            if not (isinstance(op, dict) and op.get("op") == "NOOP")
        ]
        if len(actionable) <= limit:
            return ops

        section_priority = {
            "preferences": 0,
            "decisions": 1,
            "status_changes": 2,
            "failed_attempts": 3,
            "open_questions": 4,
            "progress": 5,
            "goal": 6,
            "facts": 7,
        }
        ranked = sorted(
            enumerate(actionable),
            key=lambda item: (
                section_priority.get(item[1].get("section"), 20)
                if isinstance(item[1], dict)
                else 30,
                item[0],
            ),
        )
        keep_indexes = {index for index, _op in ranked[:limit]}
        return [op for index, op in enumerate(actionable) if index in keep_indexes]

    @staticmethod
    def _is_ordinary_non_durable_batch(
        evicted_turns: list[Turn],
        cue_re: re.Pattern[str] = STATUS_CHANGE_CUE_RE,
    ) -> bool:
        user_texts = [
            turn.content.strip()
            for turn in evicted_turns
            if turn.role == "user" and turn.content.strip()
        ]
        if not user_texts:
            return True
        combined = " ".join(user_texts)
        if DURABLE_USER_STATE_RE.search(combined):
            return False
        if any(MemoryUpdater._extract_subject_value_snippets(text) for text in user_texts):
            return False
        for text in user_texts:
            is_question = text.endswith("?") or bool(ORDINARY_QUESTION_RE.search(text))
            if cue_re.search(text) and not is_question:
                return False
        return True

    def _deterministic_subject_value_ops(
        self,
        memory: Memory,
        evicted_turns: list[Turn],
    ) -> list[dict]:
        """Conservatively preserve subject-bound dates and metrics.

        This is not the legacy exact-values inventory: generated entries go
        into normal semantic sections and keep the sentence subject attached to
        the value. Compact chat memory stores these in ``facts``; requiring a
        complete subject-bearing sentence and a durable cue prevents a bare
        numeric inventory.
        """
        if not self._has_section("facts") and not any(
            self._has_section(section) for section in ("timeline", "progress", "status_changes")
        ):
            return []

        seen_by_section = {
            section: self._active_text_keys(memory, section)
            for section in ("timeline", "progress", "status_changes", "facts")
            if self._has_section(section)
        }
        rich_sections = any(
            self._has_section(section)
            for section in ("timeline", "progress", "status_changes")
        )
        generated: list[dict] = []

        for turn in evicted_turns:
            # Assistant proposals are not user-owned state. Without an explicit
            # acceptance event, deterministic extraction must remain user-only.
            if turn.role != "user":
                continue
            snippets = self._extract_subject_value_snippets(turn.content)
            per_turn = 0
            for snippet, kind in snippets:
                if (
                    self.policy.subject_value_retention == "personal_only"
                    or not rich_sections
                ) and kind != "personal_value":
                    continue
                if (
                    self.policy.subject_value_retention == "exclude_counts"
                    and kind == "count"
                ):
                    continue
                section = self._subject_value_section(snippet, kind)
                if section is None:
                    continue
                text = self._subject_value_text(snippet, turn.role)
                if self._has_seen_text(text, seen_by_section[section]):
                    continue
                key = self._text_key(text)
                seen_by_section[section].add(key)
                op = {
                    "op": "ADD",
                    "section": section,
                    "text": text,
                    "provenance": [turn.id],
                }
                normalized = self.subject_normalizer.normalize(snippet)
                if (
                    normalized is not None
                    and normalized[0].confidence >= self.identity_confidence_threshold
                ):
                    op["subject_identity"], op["value"] = normalized
                generated.append(op)
                per_turn += 1
                if per_turn >= 6:
                    break

        return generated

    def _deterministic_exact_value_ops(
        self,
        memory: Memory,
        evicted_turns: list[Turn],
    ) -> list[dict]:
        if not self._has_section("exact_values"):
            return []

        seen = self._active_text_keys(memory, "exact_values")
        generated: list[dict] = []
        for turn in evicted_turns:
            if turn.role != "user":
                continue
            for value in self._extract_exact_values(turn.content):
                if self._has_seen_text(value, seen):
                    continue
                key = self._text_key(value)
                seen.add(key)
                generated.append(
                    {
                        "op": "ADD",
                        "section": "exact_values",
                        "text": value,
                        "provenance": [turn.id],
                    }
                )

        return generated

    def _deterministic_status_change_ops(
        self,
        memory: Memory,
        evicted_turns: list[Turn],
    ) -> list[dict]:
        if not self._has_section("status_changes"):
            return []

        seen = self._active_text_keys(memory, "status_changes")
        cue_re = status_change_cue_re(self.policy)
        generated: list[dict] = []
        for turn in evicted_turns:
            if turn.role != "user":
                continue
            snippet = self._extract_status_change_snippet(turn.content, cue_re=cue_re)
            if snippet is None and is_chat_policy(self.policy):
                snippet = self._extract_status_change_snippet(
                    turn.content,
                    cue_re=EXPLICIT_PROJECT_DENIAL_RE,
                )
            if snippet is None:
                continue
            text = f"User stated: {snippet}"
            if self._has_seen_text(text, seen):
                continue
            key = self._text_key(text)
            seen.add(key)
            generated.append(
                {
                    "op": "ADD",
                    "section": "status_changes",
                    "text": text,
                    "provenance": [turn.id],
                }
            )

        return generated

    def _drop_duplicate_deterministic_adds(self, ops: list[dict], memory: Memory) -> list[dict]:
        relevant_sections = {
            "exact_values", "facts", "preferences", "progress", "status_changes", "timeline"
        }
        active_keys_by_section: dict[str, set[str]] = {}
        filtered: list[dict] = []

        for op in ops:
            if not isinstance(op, dict):
                filtered.append(op)
                continue
            if op.get("op") != "ADD" or op.get("section") not in relevant_sections:
                filtered.append(op)
                continue

            section = op["section"]
            text = op.get("text")
            if not isinstance(text, str):
                filtered.append(op)
                continue

            if section not in active_keys_by_section:
                active_keys_by_section[section] = self._active_text_keys(memory, section)
            if self._has_seen_text(text, active_keys_by_section[section]):
                self.write_suppression_reasons["redundant_add"] += 1
                continue
            if section == "preferences":
                provenance = set(op.get("provenance") or [])
                if provenance and any(
                    entry.section == "preferences"
                    and entry.status == "active"
                    and provenance.intersection(entry.provenance)
                    for entry in memory.entries.values()
                ):
                    self.write_suppression_reasons["redundant_add"] += 1
                    continue
            filtered.append(op)

        return filtered

    def _suppress_redundant_and_transient_adds(
        self,
        ops: list[dict],
        memory: Memory,
        evicted_turns: list[Turn],
    ) -> list[dict]:
        """Drop ADDs that add no durable information to the trial memory.

        The checks deliberately require concrete evidence: an identical typed
        subject/value, an exact/contained restatement within the same section
        and source turns, or assistant-only provenance without user acceptance.
        UPDATE and SUPERSEDE operations are never changed here.
        """
        turns_by_id = {turn.id: turn for turn in evicted_turns}
        has_user_acceptance = any(
            turn.role == "user"
            and re.search(
                r"\b(?:yes|agreed|sounds good|let'?s do (?:it|that)|"
                r"go with that|accepted?)\b",
                turn.content,
                re.I,
            )
            for turn in evicted_turns
        )
        active = [entry for entry in memory.entries.values() if entry.status == "active"]
        kept: list[dict] = []

        for op in ops:
            if not isinstance(op, dict) or op.get("op") != "ADD":
                kept.append(op)
                continue

            provenance = {
                turn_id for turn_id in op.get("provenance", [])
                if isinstance(turn_id, int)
            }
            source_turns = [turns_by_id[turn_id] for turn_id in provenance if turn_id in turns_by_id]
            user_sources = [turn for turn in source_turns if turn.role == "user"]
            if source_turns and not user_sources and not has_user_acceptance:
                self.write_suppression_reasons["assistant_only_proposal"] += 1
                continue
            section = op.get("section")
            text = op.get("text")
            if not isinstance(section, str) or not isinstance(text, str):
                kept.append(op)
                continue
            op_key = self._text_key(text)
            op_identity = op.get("subject_identity")
            op_value = op.get("value")
            redundant = False
            for entry in active:
                if entry.section != section:
                    continue
                if (
                    section == "decisions"
                    and provenance
                    and provenance.issubset(set(entry.provenance))
                    and entry.text.startswith(
                        ("Accepted strategy:", "Rejected proposal:")
                    )
                ):
                    redundant = True
                    break
                if (
                    op_identity is not None
                    and op_value is not None
                    and entry.subject_identity == op_identity
                    and entry.value == op_value
                ):
                    redundant = True
                    break
                if not provenance.intersection(entry.provenance):
                    continue
                entry_key = self._text_key(entry.text)
                contained = bool(op_key and entry_key) and (
                    op_key in entry_key or entry_key in op_key
                )
                if op_key == entry_key or contained:
                    redundant = True
                    break
            if redundant:
                self.write_suppression_reasons["redundant_add"] += 1
                continue
            kept.append(op)

        return kept

    def _has_section(self, section_key: str) -> bool:
        return any(section.key == section_key for section in self.sections)

    @staticmethod
    def _active_text_keys(memory: Memory, section: str) -> set[str]:
        return {
            MemoryUpdater._text_key(entry.text)
            for entry in memory.entries.values()
            if entry.section == section and entry.status == "active"
        }

    @staticmethod
    def _extract_exact_values(content: str) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        prose = content.split("```", 1)[0]
        for pattern in EXACT_VALUE_DATE_PATTERNS:
            for match in pattern.finditer(prose):
                value = MemoryUpdater._clean_exact_value(match.group(0))
                if not value:
                    continue
                context = MemoryUpdater._date_context(prose, match.start())
                if context:
                    value = f"{context} {value}"
                if MemoryUpdater._has_seen_text(value, seen):
                    continue
                key = MemoryUpdater._text_key(value)
                seen.add(key)
                values.append(value)
        for pattern in EXACT_VALUE_PATTERNS:
            for match in pattern.finditer(prose):
                value = MemoryUpdater._clean_exact_value(match.group(0))
                if not value:
                    continue
                if MemoryUpdater._has_seen_text(value, seen):
                    continue
                key = MemoryUpdater._text_key(value)
                seen.add(key)
                values.append(value)
        return values

    @staticmethod
    def _extract_subject_value_snippets(content: str) -> list[tuple[str, str]]:
        prose = MemoryUpdater._strip_code_fences(content)
        matches: list[tuple[int, int, str]] = []
        for pattern in EXACT_VALUE_DATE_PATTERNS:
            matches.extend((match.start(), match.end(), "date") for match in pattern.finditer(prose))
        for pattern in PERSONAL_SUBJECT_VALUE_PATTERNS:
            matches.extend(
                (match.start(), match.end(), "personal_value")
                for match in pattern.finditer(prose)
            )
        for pattern in SUBJECT_VALUE_PATTERNS:
            matches.extend((match.start(), match.end(), "value") for match in pattern.finditer(prose))
        for pattern in SUBJECT_COUNT_PATTERNS:
            matches.extend((match.start(), match.end(), "count") for match in pattern.finditer(prose))

        snippets: list[tuple[str, str]] = []
        seen: set[str] = set()
        for start, end, kind in sorted(matches, key=lambda item: (item[0], item[1], item[2])):
            snippet = MemoryUpdater._snippet_around(prose, start, end)
            if not snippet or not SUBJECT_VALUE_SECTION_RE.search(snippet):
                continue
            if kind == "personal_value" and TECHNICAL_CONTEXT_RE.search(snippet):
                continue
            trimmed = trim_conversational_frame(snippet)
            if trimmed != snippet and SUBJECT_VALUE_SECTION_RE.search(trimmed):
                snippet = trimmed
            key = MemoryUpdater._text_key(f"{kind}:{snippet}")
            if key in seen:
                continue
            seen.add(key)
            snippets.append((snippet, kind))
        return snippets

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        return re.sub(r"```.*?```", " ", content, flags=re.DOTALL)

    @staticmethod
    def _snippet_around(prose: str, match_start: int, match_end: int, max_chars: int = 220) -> str:
        left_boundaries = [prose.rfind(ch, 0, match_start) for ch in ".!?\n;"]
        right_boundaries = [
            idx for idx in (prose.find(ch, match_end) for ch in ".!?\n;") if idx != -1
        ]
        left = max(left_boundaries) + 1
        right = min(right_boundaries) if right_boundaries else len(prose)
        snippet = prose[left:right].strip()

        if len(snippet) > max_chars:
            return ""

        return MemoryUpdater._clean_subject_value_snippet(snippet)

    @staticmethod
    def _clean_subject_value_snippet(snippet: str) -> str:
        snippet = re.sub(r"->->\s*[\w,/.-]+", "", snippet)
        snippet = WHITESPACE_RE.sub(" ", snippet).strip()
        snippet = snippet.strip(" -•*")
        return snippet.strip()

    def _subject_value_section(self, snippet: str, kind: str) -> str | None:
        if kind == "date" and self._has_section("timeline"):
            return "timeline"
        if (
            not is_chat_policy(self.policy)
            and STATUS_VALUE_RE.search(snippet)
            and self._has_section("status_changes")
        ):
            return "status_changes"
        if (
            not is_chat_policy(self.policy)
            and PROGRESS_VALUE_RE.search(snippet)
            and self._has_section("progress")
        ):
            return "progress"
        if self._has_section("facts"):
            return "facts"
        return None

    @staticmethod
    def _subject_value_text(snippet: str, role: str) -> str:
        prefix = "Assistant stated" if role == "assistant" else "User stated"
        return f"{prefix}: {snippet}"

    @staticmethod
    def _date_context(prose: str, match_start: int) -> str:
        """Same-sentence prefix naming what a bare date refers to."""
        boundary = max(prose.rfind(ch, 0, match_start) for ch in ".!?\n;")
        context = WHITESPACE_RE.sub(" ", prose[boundary + 1 : match_start]).strip()
        if len(context) > 70:
            context = context[-70:]
            cut = context.find(" ")
            if cut != -1:
                context = context[cut + 1 :]
        return context

    @staticmethod
    def _clean_exact_value(value: str) -> str:
        value = value.strip().strip("`")
        value = value.strip(".,;:()[]{}")
        value = WHITESPACE_RE.sub(" ", value).strip()
        if value.lower() in {"chart.js"}:
            return ""
        return value

    @staticmethod
    def _extract_status_change_snippet(
        content: str,
        cue_re: re.Pattern[str] = STATUS_CHANGE_CUE_RE,
    ) -> str | None:
        prose = content.split("```", 1)[0]
        match = cue_re.search(prose)
        if not match:
            return None

        start = max(
            prose.rfind(boundary, 0, match.start()) for boundary in (".", "\n", "。", "？", "！")
        )
        end_candidates = [
            index
            for index in (
                prose.find(".", match.end()),
                prose.find("?", match.end()),
                prose.find("!", match.end()),
                prose.find("\n", match.end()),
                prose.find("。", match.end()),
                prose.find("？", match.end()),
                prose.find("！", match.end()),
            )
            if index != -1
        ]
        start = 0 if start == -1 else start + 1
        end = min(end_candidates) + 1 if end_candidates else len(prose)
        snippet = WHITESPACE_RE.sub(" ", prose[start:end]).strip()
        snippet = snippet.rstrip(" ->")
        if not snippet:
            return None
        trimmed = trim_conversational_frame(snippet)
        if cue_re.search(trimmed):
            snippet = trimmed
        if len(snippet) > 500:
            return None
        return snippet

    @staticmethod
    def _text_key(text: str) -> str:
        return WHITESPACE_RE.sub(" ", text).strip().lower()

    @staticmethod
    def _has_seen_text(text: str, seen: set[str]) -> bool:
        key = MemoryUpdater._text_key(text)
        return any(key == old or key in old or old in key for old in seen)

    @staticmethod
    def _normalize_text_provenance(op: dict) -> None:
        provenance = op.get("provenance")
        if isinstance(provenance, list):
            op["provenance"] = [
                int(value) if isinstance(value, str) and value.isdigit() else value
                for value in provenance
            ]
        text = op.get("text")
        if not isinstance(text, str):
            return

        match = _TURN_SUFFIX_RE.search(text)
        if not match:
            return

        turn_ids: list[int] = []
        for part in match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = [value.strip() for value in part.split("-", 1)]
                if start_text.isdigit() and end_text.isdigit():
                    start, end = int(start_text), int(end_text)
                    if start <= end:
                        turn_ids.extend(range(start, end + 1))
                    else:
                        turn_ids.extend(range(end, start + 1))
                continue
            if part.isdigit():
                turn_ids.append(int(part))

        if turn_ids:
            provenance = op.get("provenance")
            if not isinstance(provenance, list) or not provenance:
                op["provenance"] = sorted(set(turn_ids))
            elif all(isinstance(turn_id, int) for turn_id in provenance):
                op["provenance"] = sorted(set(provenance) | set(turn_ids))

        op["text"] = _TURN_SUFFIX_RE.sub("", text).strip()

    @staticmethod
    def _normalize_entry_id(entry_id: object, memory: Memory) -> str | None:
        if isinstance(entry_id, str):
            if entry_id in memory.entries:
                return entry_id
        return None

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

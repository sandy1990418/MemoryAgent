"""LLM-driven updater that turns evicted turns into memory operations."""

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
from memory_agent.models.policy import (
    AGENT_POLICY,
    MemoryPolicy,
    is_chat_policy,
    validate_policy_sections,
)
from memory_agent.models.sections import SectionConfig
from memory_agent.models.transcript import Turn
from memory_agent.models.memory import SubjectNormalizer
from memory_agent.profiles.chat.subject_normalizer import ChatSubjectNormalizer
from memory_agent.structured.memory import Memory
from memory_agent.structured.heuristics import (
    ASSISTANT_ATTRIBUTED_RE,
    DURABLE_USER_STATE_RE,
    EXACT_VALUE_DATE_PATTERNS,
    EXACT_VALUE_PATTERNS,
    EXPLICIT_PROJECT_DENIAL_RE,
    GENERIC_NON_DURABLE_MEMORY_RE,
    ORDINARY_QUESTION_RE,
    PROGRESS_VALUE_RE,
    PROJECT_IMPLEMENTATION_STATE_RE,
    STATUS_CHANGE_CUE_RE,
    STATUS_VALUE_RE,
    STABLE_INSTRUCTION_RE,
    SUBJECT_VALUE_PATTERNS,
    SUBJECT_VALUE_SECTION_RE,
    WHITESPACE_RE,
    content_words,
    status_change_cue_re,
)
from memory_agent.structured.ops import UpdateFailed, parse_memory_ops
from memory_agent.structured.prompts import build_updater_prompt
from memory_agent.structured.update_selector import UpdateMemorySelector

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

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def input_tokens(self) -> int:
        return self.system_tokens + self.visible_memory_tokens + self.evicted_turn_tokens + self.retry_tokens


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
        policy: MemoryPolicy | None = None,
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
        self.evicted_user_assistant_pairs = 0
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
            "retries": sum(1 for r in self.token_reports if r.retry_tokens),
            "retry_tokens": sum(r.retry_tokens for r in self.token_reports),
            "rejected_ops_count": sum(r.rejected_ops_count for r in self.token_reports),
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
        if self.evicted_turn_token_budget is None:
            return turns
        selected: list[Turn] = []
        used = 0
        # Keep the newest complete turns; never slice turn text mid-content.
        for turn in reversed(turns):
            tokens = self.token_estimator(turn.content)
            if used + tokens > self.evicted_turn_token_budget:
                continue
            selected.append(turn)
            used += tokens
        return list(reversed(selected))

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
        if re.search(r"\b(?:yes|agreed|sounds good|let'?s do (?:it|that)|go with that|accepted?)\b", combined, re.I):
            return "call:user_acceptance_ambiguous"
        if re.search(r"\b(?:correction|changed my mind|no longer|not anymore|instead|actually|contradict(?:s|ion|ory)?)\b", combined, re.I):
            return "call:unresolved_subject_conflict"
        if self._is_ordinary_non_durable_batch(
            evicted_turns, cue_re=status_change_cue_re(self.policy)
        ):
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
                *self._consolidate_near_duplicates(memory),
                *self._consolidate_latest_subject_values(
                    memory, self.subject_normalizer, self.identity_confidence_threshold
                ),
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
                raise UpdateFailed(f"LLM transport error: {exc}") from exc

            ops = parse_memory_ops(response)
            if ops is None:
                raise UpdateFailed(f"Could not parse a JSON ops array from LLM response: {response!r}")
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
            _debug_ops("llm after filter", ops)
            ops = [
                op for op in ops if not (isinstance(op, dict) and op.get("op") == "NOOP")
            ]
            if not ops:
                consolidated = [
                    *self._consolidate_near_duplicates(memory),
                    *self._consolidate_latest_subject_values(
                        memory, self.subject_normalizer, self.identity_confidence_threshold
                    ),
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
                    *self._consolidate_near_duplicates(memory),
                    *self._consolidate_latest_subject_values(
                        memory, self.subject_normalizer, self.identity_confidence_threshold
                    ),
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
        """Keep the latest value only for confidently identical typed subjects."""
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
            key = (identity.namespace, identity.entity, identity.attribute, identity.qualifier, value.unit)
            groups.setdefault(key, []).append(entry)
        for matches in groups.values():
            if len(matches) < 2:
                continue
            keep = max(matches, key=lambda entry: (max(entry.provenance or [0]), entry.id))
            provenance = sorted({turn_id for entry in matches for turn_id in entry.provenance})
            ops.append({
                "op": "UPDATE",
                "id": keep.id,
                "text": keep.text,
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
        ops.extend(self._deterministic_preference_ops(memory, evicted_turns))
        ops.extend(self._deterministic_project_state_ops(memory, evicted_turns))
        if self.policy.allow_exact_values:
            ops.extend(self._deterministic_exact_value_ops(memory, evicted_turns))
        if self.policy.allow_deterministic_subject_values:
            ops.extend(self._deterministic_subject_value_ops(memory, evicted_turns))
        ops.extend(self._deterministic_status_change_ops(memory, evicted_turns))
        return ops

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
        for turn in evicted_turns:
            if turn.role != "user":
                continue
            prose = self._strip_code_fences(turn.content).split("->->", 1)[0]
            for sentence in re.split(r"(?<=[.!?])\s+|\n", prose):
                sentence = WHITESPACE_RE.sub(" ", sentence).strip()
                if not sentence or not PROJECT_IMPLEMENTATION_STATE_RE.search(sentence):
                    continue
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
                if len(generated) >= 2:
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
        disallowed_sections = {"timeline", "tool_facts", "exact_values", "progress"}
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
                canonical = self._canonical_chat_entry_text(text, section)
                if canonical is None:
                    continue
                op["text"] = canonical
                explicit_denial = section == "status_changes" and bool(
                    EXPLICIT_PROJECT_DENIAL_RE.search(text)
                )
                if ordinary_question and not explicit_denial:
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
        the value. It is enabled only for richer agent memory configs that have
        timeline/progress/status-change sections, so simple chat memory does
        not become a numeric scrape.
        """
        if not any(
            self._has_section(section)
            for section in ("timeline", "progress", "status_changes")
        ):
            return []

        seen_by_section = {
            section: self._active_text_keys(memory, section)
            for section in ("timeline", "progress", "status_changes", "facts")
            if self._has_section(section)
        }
        generated: list[dict] = []

        for turn in evicted_turns:
            # Assistant proposals are not user-owned state. Without an explicit
            # acceptance event, deterministic extraction must remain user-only.
            if turn.role != "user":
                continue
            snippets = self._extract_subject_value_snippets(turn.content)
            per_turn = 0
            for snippet, kind in snippets:
                section = self._subject_value_section(snippet, kind)
                if section is None:
                    continue
                text = self._subject_value_text(snippet, turn.role)
                if self._has_seen_text(text, seen_by_section[section]):
                    continue
                key = self._text_key(text)
                seen_by_section[section].add(key)
                generated.append(
                    {
                        "op": "ADD",
                        "section": section,
                        "text": text,
                        "provenance": [turn.id],
                    }
                )
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
                continue
            if section == "preferences":
                provenance = set(op.get("provenance") or [])
                if provenance and any(
                    entry.section == "preferences"
                    and entry.status == "active"
                    and provenance.intersection(entry.provenance)
                    for entry in memory.entries.values()
                ):
                    continue
            filtered.append(op)

        return filtered

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
        for pattern in SUBJECT_VALUE_PATTERNS:
            matches.extend((match.start(), match.end(), "value") for match in pattern.finditer(prose))

        snippets: list[tuple[str, str]] = []
        seen: set[str] = set()
        for start, end, kind in sorted(matches, key=lambda item: (item[0], item[1], item[2])):
            snippet = MemoryUpdater._snippet_around(prose, start, end)
            if not snippet or not SUBJECT_VALUE_SECTION_RE.search(snippet):
                continue
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
        if PROGRESS_VALUE_RE.search(snippet) and self._has_section("progress"):
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

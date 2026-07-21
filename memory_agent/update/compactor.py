"""Subject-aware compaction for structured memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from memory_agent.clients.llm import LLMClient
from memory_agent.core.models import MemoryEntry
from memory_agent.core.sections import SectionConfig
from memory_agent.core.store import Memory
from memory_agent.policies.structured import (
    CHAT_POLICY,
    StructuredMemoryPolicy,
    validate_policy_sections,
)
from memory_agent.update.operations import parse_memory_ops
from memory_agent.update.prompts import build_compactor_prompt


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class CompactionCandidate:
    """A structurally bounded active-entry batch for LLM review."""

    subject_key: str
    entries: tuple[MemoryEntry, ...]
    reason: str


@dataclass
class CompactionMetrics:
    attempted_calls: int = 0
    successful_compactions: int = 0
    failed_compactions: int = 0
    rejected_compactions: int = 0
    skipped_compactions: int = 0
    before_active: int = 0
    after_active: int = 0
    candidate_tokens: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    candidate_results: list[dict[str, object]] = field(default_factory=list)

    def record_reason(self, reason: str) -> None:
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1


class MemoryCompactor:
    """Ask the LLM to compact bounded entry batches without local semantics."""

    def __init__(
        self,
        llm: LLMClient,
        sections: list[SectionConfig],
        policy: StructuredMemoryPolicy | None = None,
        model: str | None = None,
        token_estimator: Callable[[str], int] | None = None,
        max_candidate_entries: int = 8,
        max_candidate_tokens: int = 512,
    ) -> None:
        self.llm = llm
        self.sections = sections
        self.policy = policy or CHAT_POLICY
        validate_policy_sections(self.policy, sections)
        self.model = model
        self.token_estimator = token_estimator or _default_token_estimator
        self.max_candidate_entries = max(2, max_candidate_entries)
        self.max_candidate_tokens = max_candidate_tokens
        self.metrics = CompactionMetrics()
        self._section_keys = {section.key for section in sections}

    def record_skip(self, reason: str) -> None:
        """Record a protective skip that intentionally avoids transport."""
        self.metrics.skipped_compactions += 1
        self.metrics.record_reason(reason)

    def _build_prompt(
        self, memory: Memory, candidate: CompactionCandidate | None = None
    ) -> tuple[str, list[dict]]:
        entries = list(candidate.entries) if candidate is not None else []
        rendered = memory.render(entries=entries) or "(No candidate entries.)"
        return build_compactor_prompt(
            sections=self.sections,
            current_memory=rendered,
        )

    def detect_candidates(self, memory: Memory) -> list[CompactionCandidate]:
        """Build deterministic section/recency chunks for LLM review only."""
        active = sorted(
            (entry for entry in memory.entries.values() if entry.status == "active"),
            key=lambda entry: (entry.section, max(entry.provenance or [-1]), entry.id),
        )
        candidates: list[CompactionCandidate] = []
        for section in sorted(self._section_keys):
            entries = [entry for entry in active if entry.section == section]
            for offset in range(0, len(entries), self.max_candidate_entries):
                chunk = tuple(entries[offset:offset + self.max_candidate_entries])
                if len(chunk) < 2:
                    continue
                candidates.append(CompactionCandidate(
                    subject_key=f"{section}:{offset}",
                    entries=chunk,
                    reason="llm-candidate",
                ))
        return candidates

    def compact(self, memory: Memory) -> tuple[list[dict], list[dict]]:
        """Compatibility entry point: compact detected bounded candidates only."""
        return self.compact_candidates(memory, self.detect_candidates(memory))

    def compact_candidates(
        self, memory: Memory, candidates: list[CompactionCandidate]
    ) -> tuple[list[dict], list[dict]]:
        all_applied: list[dict] = []
        all_rejected: list[dict] = []
        self.metrics.before_active = sum(e.status == "active" for e in memory.entries.values())
        for candidate in candidates:
            before = sum(e.status == "active" for e in memory.entries.values())
            attempted_before = self.metrics.attempted_calls
            applied, rejected = self._compact_candidate(memory, candidate)
            after = sum(e.status == "active" for e in memory.entries.values())
            if rejected:
                outcome = "rejected" if rejected[0].get("reason") != "transport" else "failed"
            elif self.metrics.attempted_calls > attempted_before and applied:
                outcome = "successful"
            else:
                outcome = "skipped"
            self.metrics.candidate_results.append({
                "subject_key": candidate.subject_key,
                "before_active": before,
                "after_active": after,
                "outcome": outcome,
            })
            all_applied.extend(applied)
            all_rejected.extend(rejected)
        self.metrics.after_active = sum(e.status == "active" for e in memory.entries.values())
        return all_applied, all_rejected

    def _compact_candidate(
        self, memory: Memory, candidate: CompactionCandidate
    ) -> tuple[list[dict], list[dict]]:
        visible_ids = {entry.id for entry in candidate.entries}
        if len(candidate.entries) > self.max_candidate_entries:
            return self._reject(candidate, "budget")
        rendered = memory.render(entries=list(candidate.entries))
        tokens = self.token_estimator(rendered)
        candidate_token_limit = self.max_candidate_tokens
        if tokens > candidate_token_limit:
            return self._reject(candidate, "budget")
        self.metrics.candidate_tokens += tokens

        system, messages = self._build_prompt(memory, candidate)
        try:
            self.metrics.attempted_calls += 1
            response = self.llm.complete(system, messages, model=self.model)
        except Exception as exc:
            self.metrics.failed_compactions += 1
            self.metrics.record_reason("transport")
            return [], [{"candidate": candidate.subject_key, "reason": "transport", "detail": str(exc)}]

        ops = parse_memory_ops(response)
        if ops is None:
            return self._reject(candidate, "schema")
        ops = self._normalize_ops(ops)
        ops = [
            op for op in ops if not (isinstance(op, dict) and op.get("op") == "NOOP")
        ]
        if not ops:
            self.metrics.skipped_compactions += 1
            return [], []
        applied, rejected = self._apply_candidate_ops(memory, candidate, ops, visible_ids)
        if not rejected:
            self.metrics.successful_compactions += 1
        return applied, rejected


    def _reject(
        self,
        candidate: CompactionCandidate,
        reason: str,
        detail: object | None = None,
    ):
        self.metrics.rejected_compactions += 1
        self.metrics.record_reason(reason)
        rejection = {"candidate": candidate.subject_key, "reason": reason}
        if detail is not None:
            rejection["detail"] = detail
        return [], [rejection]

    def _apply_candidate_ops(self, memory, candidate, ops, visible_ids):
        referenced = {op.get("id") for op in ops if isinstance(op, dict) and op.get("op") == "SUPERSEDE"}
        if not referenced.issubset(visible_ids):
            return self._reject(
                candidate,
                "hidden_id",
                {
                    "unexpected_ids": sorted(referenced - visible_ids),
                    "visible_ids": sorted(visible_ids),
                },
            )
        rejected = self._validate_ops(memory, ops)
        if rejected:
            reason = "provenance" if any("provenance" in item["reason"] for item in rejected) else "schema"
            return self._reject(candidate, reason, rejected)
        before = sum(memory.entries[entry_id].status == "active" for entry_id in visible_ids)
        trial = memory._copy()
        applied, rejected = trial.apply_ops_atomically(ops)
        if rejected:
            return self._reject(candidate, "schema")
        after = sum(entry.status == "active" for entry in trial.entries.values())
        total_before = sum(entry.status == "active" for entry in memory.entries.values())
        if after >= total_before or before < 2:
            return self._reject(candidate, "no_reduction")
        memory.entries, memory.narrative, memory._counters = trial.entries, trial.narrative, trial._counters
        return applied, []

    @staticmethod
    def _normalize_ops(ops: list[dict]) -> list[dict]:
        """Normalize operation spelling without translating legacy schemas."""
        return [
            {**op, "op": str(op.get("op", "")).upper()}
            if isinstance(op, dict)
            else op
            for op in ops
        ]

    def _validate_ops(self, memory: Memory, ops: list[dict]) -> list[dict]:
        rejected: list[dict] = []
        supersede_ids: set[str] = set()
        adds: list[dict] = []

        for op in ops:
            if not isinstance(op, dict):
                rejected.append({"op": op, "reason": "op is not a dict"})
                continue
            kind = op.get("op")
            if kind == "SUPERSEDE":
                entry_id = op.get("id")
                entry = memory.entries.get(entry_id) if isinstance(entry_id, str) else None
                if entry is None:
                    rejected.append({"op": op, "reason": "unknown memory entry id"})
                elif entry.status != "active":
                    rejected.append(
                        {"op": op, "reason": "cannot compact a superseded entry"}
                    )
                elif entry_id in supersede_ids:
                    rejected.append({"op": op, "reason": "duplicate SUPERSEDE id"})
                else:
                    supersede_ids.add(entry_id)
            elif kind == "ADD":
                section = op.get("section")
                text = op.get("text")
                provenance = op.get("provenance")
                if section not in self._section_keys:
                    rejected.append({"op": op, "reason": f"unknown section: {section}"})
                elif not isinstance(text, str) or not text.strip():
                    rejected.append({"op": op, "reason": "missing/invalid text"})
                elif (
                    not isinstance(provenance, list)
                    or not provenance
                    or any(not isinstance(turn_id, int) for turn_id in provenance)
                ):
                    rejected.append({"op": op, "reason": "invalid provenance"})
                else:
                    adds.append(op)
            else:
                rejected.append(
                    {"op": op, "reason": "compaction only accepts ADD and SUPERSEDE"}
                )

        if rejected:
            return rejected
        if not supersede_ids or not adds:
            return [
                {
                    "op": ops,
                    "reason": "compaction requires replaced entries and canonical ADDs",
                }
            ]

        affected_sections = {
            memory.entries[entry_id].section for entry_id in supersede_ids
        }
        add_sections = {op["section"] for op in adds}
        protected_sections = {
            "preferences",
            "decisions",
            "failed_attempts",
            "open_questions",
            "progress",
        }
        missing_sections = (affected_sections & protected_sections) - add_sections
        if missing_sections:
            return [
                {
                    "op": ops,
                    "reason": (
                        "canonical ADD missing for affected sections: "
                        + ", ".join(sorted(missing_sections))
                    ),
                }
            ]

        source_provenance = {
            turn_id
            for entry_id in supersede_ids
            for turn_id in memory.entries[entry_id].provenance
        }
        canonical_provenance = {
            turn_id for op in adds for turn_id in op.get("provenance", [])
        }
        if not source_provenance.issubset(canonical_provenance):
            return [
                {
                    "op": adds,
                    "reason": "canonical provenance must preserve all source turn ids",
                }
            ]
        return []

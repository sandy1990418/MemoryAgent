"""Subject-aware compaction for structured memory."""

from __future__ import annotations

import json
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
        rendered = self._render_candidate(memory, entries)
        return build_compactor_prompt(
            sections=self.sections,
            current_memory=rendered,
        )

    @staticmethod
    def _render_candidate(memory: Memory, entries: list[MemoryEntry]) -> str:
        """Render candidate text together with the canonical source fields.

        The normal memory renderer intentionally omits provenance from user
        context. Compaction needs those structural ids to produce a canonical
        replacement, so expose them in a bounded, machine-readable block next
        to the human-readable entry text. Keeping this at the compactor
        boundary avoids changing the public memory-rendering contract.
        """
        rendered = memory.render(entries=entries) or "(No candidate entries.)"
        if not entries:
            return rendered
        metadata = [
            {
                "id": entry.id,
                "section": entry.section,
                "text": entry.text,
                "provenance": list(entry.provenance),
            }
            for entry in entries
        ]
        return (
            f"{rendered}\n\n"
            "Canonical candidate entries (copy these source fields exactly):\n"
            f"{json.dumps(metadata, ensure_ascii=False, indent=2)}"
        )

    def detect_candidates(self, memory: Memory) -> list[CompactionCandidate]:
        """Build deterministic, overlapping section/recency windows.

        A non-overlapping partition makes entries on opposite sides of a
        chunk boundary impossible for the model to compare.  Sliding each
        bounded window by one entry keeps the review structurally bounded
        while giving adjacent windows one shared anchor.  The overlap is
        intentionally structural; no local similarity or lexical matching is
        performed here.
        """
        active = sorted(
            (entry for entry in memory.entries.values() if entry.status == "active"),
            key=lambda entry: (entry.section, max(entry.provenance or [-1]), entry.id),
        )
        candidates: list[CompactionCandidate] = []
        step = max(1, self.max_candidate_entries - 1)
        for section in sorted(self._section_keys):
            entries = [entry for entry in active if entry.section == section]
            offsets = [0]
            while offsets[-1] + self.max_candidate_entries < len(entries):
                next_offset = offsets[-1] + step
                # Always review a bounded tail window.  This avoids ending on
                # a tiny suffix after a boundary-overlap window and gives the
                # final active state a full opportunity for consolidation.
                if next_offset + self.max_candidate_entries >= len(entries):
                    tail_offset = len(entries) - self.max_candidate_entries
                    if tail_offset != offsets[-1]:
                        offsets.append(tail_offset)
                    break
                offsets.append(next_offset)
            for offset in offsets:
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
        # Candidates may overlap.  An earlier window can supersede one of the
        # shared entries, so review only the active subset at the time this
        # window is processed and avoid presenting stale ids to the model.
        active_entries = tuple(
            entry
            for entry in candidate.entries
            if (current := memory.entries.get(entry.id)) is not None
            and current.status == "active"
        )
        if len(active_entries) < 2:
            self.metrics.skipped_compactions += 1
            self.metrics.record_reason("stale_candidate")
            return [], []
        candidate = CompactionCandidate(
            subject_key=candidate.subject_key,
            entries=active_entries,
            reason=candidate.reason,
        )
        visible_ids = {entry.id for entry in candidate.entries}
        if len(candidate.entries) > self.max_candidate_entries:
            return self._reject(candidate, "budget")
        rendered = self._render_candidate(memory, list(candidate.entries))
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
        rejection: dict[str, object] = {
            "candidate": candidate.subject_key,
            "reason": reason,
        }
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
        rejected = self._validate_ops(memory, ops, visible_ids=visible_ids)
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

    def _validate_ops(
        self,
        memory: Memory,
        ops: list[dict],
        *,
        visible_ids: set[str] | None = None,
    ) -> list[dict]:
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
                elif isinstance(entry_id, str):
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
                elif self._contains_bookkeeping(text, visible_ids or set()):
                    rejected.append(
                        {
                            "op": op,
                            "reason": (
                                "canonical text contains entry ids or compaction "
                                "bookkeeping"
                            ),
                        }
                    )
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
        missing_sections = affected_sections - add_sections
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

    @staticmethod
    def _contains_bookkeeping(text: str, source_ids: set[str]) -> bool:
        """Reject machine-facing metadata copied into user-visible text.

        This is deliberately a structural guard, not semantic extraction:
        exact source entry ids and explicit operation/metadata markers are
        never valid parts of a canonical memory sentence.  Tokenizing only on
        punctuation avoids treating an id such as ``F1`` as a substring of a
        normal word while catching common forms such as ``[F1]`` and ``F1/F2``.
        """
        tokens = text
        for separator in "[](){}:,;.!?/\\\"'`\n\r\t":
            tokens = tokens.replace(separator, " ")
        if any(token in source_ids for token in tokens.split()):
            return True
        folded = text.casefold()
        return any(
            marker in text or marker.casefold() in folded
            for marker in (
                "SUPERSEDE",
                "NOOP",
                '"op"',
                '"id"',
                '"provenance"',
                "provenance:",
                "source entries",
                "entry ids",
            )
        )

"""Subject-aware compaction for structured memory."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable

from memory_agent.clients.llm import LLMClient
from memory_agent.core.models import MemoryEntry
from memory_agent.core.sections import SectionConfig
from memory_agent.core.store import Memory
from memory_agent.policies.structured import (
    StructuredMemoryPolicy,
    get_memory_policy,
    validate_policy_sections,
)
from memory_agent.update.operations import parse_memory_ops
from memory_agent.update.prompts import build_compactor_prompt


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class CompactionCandidate:
    """A bounded active-entry cluster safe to consider in isolation."""

    subject_key: str
    entries: tuple[MemoryEntry, ...]
    reason: str


@dataclass
class CompactionMetrics:
    attempted_calls: int = 0
    successful_compactions: int = 0
    deterministic_compactions: int = 0
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


_WORDS_RE = re.compile(r"[a-z0-9]+")
_VALUE_RE = re.compile(r"^(?:\d+(?:\.\d+)?(?:ms|s|%|gb|mb)?|now|was|is|the|a|an|for|to|use|uses)$")


class MemoryCompactor:
    """Merge active entries by subject without deleting historical entries."""

    def __init__(
        self,
        llm: LLMClient,
        sections: list[SectionConfig],
        policy: StructuredMemoryPolicy | None = None,
        model: str | None = None,
        max_memory_tokens: int | None = None,
        token_estimator: Callable[[str], int] | None = None,
        max_candidate_entries: int = 8,
        max_candidate_tokens: int = 512,
        enable_semantic_candidates: bool = True,
    ) -> None:
        self.llm = llm
        self.sections = sections
        self.policy = policy or get_memory_policy(None)
        validate_policy_sections(self.policy, sections)
        self.model = model
        self.max_memory_tokens = max_memory_tokens
        self.token_estimator = token_estimator or _default_token_estimator
        self.max_candidate_entries = max_candidate_entries
        self.max_candidate_tokens = max_candidate_tokens
        self.enable_semantic_candidates = enable_semantic_candidates
        self.metrics = CompactionMetrics()
        self._section_keys = {section.key for section in sections}
        self._section_key_by_prefix = {
            section.prefix.lower(): section.key for section in sections
        }

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

    @staticmethod
    def _identity_key(entry: MemoryEntry) -> str | None:
        identity = entry.subject_identity
        if identity is None or identity.confidence < 0.8:
            return None
        entity = identity.entity.strip().lower()
        attribute = identity.attribute.strip().lower()
        if not entity or entity == attribute or entity in {
            "goal", "target", "budget", "rate", "duration", "value",
            "my goal", "our goal", "the goal", "a goal",
        }:
            return None
        unit = entry.value.unit if entry.value is not None else None
        # Empty qualifier/unit are identity components too; omitting them would
        # collapse qualified and unqualified facts or unit-bearing values.
        return "|".join(
            value if value is not None else "<none>" for value in (
                identity.namespace, identity.entity, identity.attribute,
                identity.qualifier, unit,
            )
        )

    @staticmethod
    def _lexical_key(entry: MemoryEntry) -> str:
        words = [word for word in _WORDS_RE.findall(entry.text.lower()) if not _VALUE_RE.match(word)]
        return f"{entry.section}:" + " ".join(sorted(set(words)))

    def detect_candidates(self, memory: Memory) -> list[CompactionCandidate]:
        """Detect deterministic, bounded active clusters without reading history."""
        active = [entry for entry in memory.entries.values() if entry.status == "active"]
        groups: dict[str, list[MemoryEntry]] = {}
        reasons: dict[str, str] = {}
        for entry in active:
            identity_key = self._identity_key(entry)
            text_key = " ".join(entry.text.lower().split())
            key = identity_key or f"exact:{entry.section}:{text_key}"
            groups.setdefault(key, []).append(entry)
            reasons[key] = "typed-subject" if identity_key else "exact-duplicate"

        if not self.enable_semantic_candidates:
            return self._candidates_from_groups(memory, groups, reasons)

        # Conservative semantic clusters are LLM-only: require same section and
        # meaningful token overlap, and never mix a typed group with another subject.
        untyped = [
            entry
            for entry in active
            if entry.subject_identity is None and self._identity_key(entry) is None
        ]
        for index, left in enumerate(untyped):
            left_words = set(self._lexical_key(left).split(":", 1)[1].split())
            for right in untyped[index + 1:]:
                if left.section != right.section:
                    continue
                right_words = set(self._lexical_key(right).split(":", 1)[1].split())
                if left_words & right_words:
                    key = f"semantic:{left.section}:{min(left.id, right.id)}"
                    groups.setdefault(key, []).extend((left, right))
                    reasons[key] = "semantic-overlap"

        return self._candidates_from_groups(memory, groups, reasons)

    @staticmethod
    def _candidates_from_groups(
        memory: Memory,
        groups: dict[str, list[MemoryEntry]],
        reasons: dict[str, str],
    ) -> list[CompactionCandidate]:
        candidates: list[CompactionCandidate] = []
        seen_sets: set[frozenset[str]] = set()
        for key, entries in groups.items():
            unique = tuple(dict.fromkeys(entry.id for entry in entries))
            if len(unique) < 2:
                continue
            entry_set = frozenset(unique)
            if entry_set in seen_sets:
                continue
            seen_sets.add(entry_set)
            selected = tuple(memory.entries[entry_id] for entry_id in sorted(unique))
            candidates.append(CompactionCandidate(key, selected, reasons[key]))
        return sorted(candidates, key=lambda item: item.subject_key)

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
            deterministic_before = self.metrics.deterministic_compactions
            applied, rejected = self._compact_candidate(memory, candidate)
            after = sum(e.status == "active" for e in memory.entries.values())
            if rejected:
                outcome = "rejected" if rejected[0].get("reason") != "transport" else "failed"
            elif self.metrics.deterministic_compactions > deterministic_before:
                outcome = "deterministic"
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
        if tokens > self.max_candidate_tokens:
            return self._reject(candidate, "budget")
        self.metrics.candidate_tokens += tokens

        texts = {" ".join(entry.text.lower().split()) for entry in candidate.entries}
        identity_keys = [self._identity_key(entry) for entry in candidate.entries]
        typed = identity_keys[0] is not None and len(set(identity_keys)) == 1
        if len(texts) == 1 or typed:
            latest = max(candidate.entries, key=lambda entry: (max(entry.provenance, default=-1), entry.id))
            provenance = sorted({turn for entry in candidate.entries for turn in entry.provenance})
            ops = [
                {"op": "SUPERSEDE", "id": entry.id, "reason": "Deterministic subject compaction."}
                for entry in candidate.entries
            ] + [{"op": "ADD", "section": latest.section, "text": latest.text,
                  "provenance": provenance, "subject_identity": latest.subject_identity,
                  "value": latest.value}]
            applied, rejected = self._apply_candidate_ops(memory, candidate, ops, visible_ids)
            if rejected:
                return applied, rejected
            self.metrics.deterministic_compactions += 1
            return applied, []

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
        ops = self._normalize_ops(ops, memory)
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

    def _reject(self, candidate: CompactionCandidate, reason: str):
        self.metrics.rejected_compactions += 1
        self.metrics.record_reason(reason)
        return [], [{"candidate": candidate.subject_key, "reason": reason}]

    def _apply_candidate_ops(self, memory, candidate, ops, visible_ids):
        referenced = {op.get("id") for op in ops if isinstance(op, dict) and op.get("op") == "SUPERSEDE"}
        if not referenced.issubset(visible_ids):
            return self._reject(candidate, "hidden_id")
        rejected = self._validate_ops(memory, ops)
        if rejected:
            reason = "provenance" if any("provenance" in item["reason"] for item in rejected) else "schema"
            return self._reject(candidate, reason)
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

    def _normalize_ops(self, ops: list[dict], memory: Memory) -> list[dict]:
        normalized: list[dict] = []
        for op in ops:
            if isinstance(op, str) and op.strip().upper() == "NOOP":
                normalized.append({"op": "NOOP"})
                continue
            if not isinstance(op, dict):
                normalized.append(op)
                continue
            item = dict(op)
            kind = str(item.get("op", "")).upper()
            item["op"] = kind
            source_ids = next(
                (
                    item.pop(key)
                    for key in (
                        "source_provenance_ids",
                        "sourceProvenanceIds",
                        "source_provenance",
                        "provenance_ids",
                        "canonicalAddProvenanceIds",
                    )
                    if key in item
                ),
                None,
            )
            if source_ids is None and isinstance(item.get("provenance"), list) and all(
                isinstance(value, str) for value in item["provenance"]
            ):
                source_ids = item.pop("provenance")
            if kind == "SUPERSEDE" and isinstance(source_ids, list):
                for entry_id in source_ids:
                    normalized.append({
                        "op": "SUPERSEDE",
                        "id": entry_id,
                        "reason": item.get("reason", "Consolidated by subject."),
                    })
                continue
            if item.get("op") == "ADD":
                item.pop("id", None)
                section = item.get("section", item.pop("key", None))
                if isinstance(section, str):
                    item["section"] = self._section_key_by_prefix.get(
                        section.lower(),
                        section,
                    )
                value = item.pop("value", None)
                if "text" not in item and isinstance(value, str):
                    item["text"] = value
                elif "text" not in item and isinstance(value, dict):
                    item["text"] = next(
                        (
                            value[key]
                            for key in ("details", "description", "currentTruth", "content")
                            if isinstance(value.get(key), str)
                        ),
                        "",
                    )
                if "text" not in item and isinstance(item.get("currentTruth"), str):
                    item["text"] = item.pop("currentTruth")
                if "provenance" not in item and isinstance(source_ids, list):
                    item["provenance"] = sorted({
                        turn_id
                        for entry_id in source_ids
                        for turn_id in (
                            memory.entries[entry_id].provenance
                            if entry_id in memory.entries
                            else []
                        )
                    })
            normalized.append(item)
        return normalized

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

        historical_texts = {
            self._text_key(entry.text)
            for entry in memory.entries.values()
            if entry.status == "superseded"
        }
        reactivated = [
            op for op in adds if self._text_key(op["text"]) in historical_texts
        ]
        if reactivated:
            return [
                {
                    "op": reactivated,
                    "reason": "canonical ADD would re-activate superseded content",
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
    def _text_key(text: str) -> str:
        return " ".join(text.lower().split())

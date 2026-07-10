"""Subject-based compaction for structured memory."""

from __future__ import annotations

from typing import Callable

from memory_agent.clients.llm import LLMClient
from memory_agent.models.policy import MemoryPolicy, get_memory_policy, validate_policy_sections
from memory_agent.models.sections import SectionConfig
from memory_agent.structured.memory import Memory
from memory_agent.structured.ops import UpdateFailed, parse_memory_ops
from memory_agent.structured.prompts import build_compactor_prompt


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


class MemoryCompactor:
    """Merge active entries by subject without deleting historical entries."""

    def __init__(
        self,
        llm: LLMClient,
        sections: list[SectionConfig],
        policy: MemoryPolicy | None = None,
        model: str | None = None,
        max_memory_tokens: int | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        self.llm = llm
        self.sections = sections
        self.policy = policy or get_memory_policy(None)
        validate_policy_sections(self.policy, sections)
        self.model = model
        self.max_memory_tokens = max_memory_tokens
        self.token_estimator = token_estimator or _default_token_estimator
        self._section_keys = {section.key for section in sections}
        self._section_key_by_prefix = {
            section.prefix.lower(): section.key for section in sections
        }

    def _build_prompt(self, memory: Memory) -> tuple[str, list[dict]]:
        rendered = memory.render(
            include_superseded=True,
            max_tokens=self.max_memory_tokens,
            token_estimator=self.token_estimator,
        ) or "(No memory entries yet.)"
        return build_compactor_prompt(
            sections=self.sections,
            current_memory=rendered,
        )

    def compact(self, memory: Memory) -> tuple[list[dict], list[dict]]:
        """Generate, validate, and atomically apply compaction operations."""
        system, messages = self._build_prompt(memory)
        try:
            response = self.llm.complete(system, messages, model=self.model)
        except Exception as exc:
            raise UpdateFailed(f"Compactor LLM transport error: {exc}") from exc

        ops = parse_memory_ops(response)
        if ops is None:
            raise UpdateFailed(
                f"Could not parse a JSON compaction ops array from LLM response: {response!r}"
            )
        ops = self._normalize_ops(ops)
        ops = [
            op for op in ops if not (isinstance(op, dict) and op.get("op") == "NOOP")
        ]
        if not ops:
            return [], []

        rejected = self._validate_ops(memory, ops)
        if rejected:
            return [], rejected

        before_active = sum(entry.status == "active" for entry in memory.entries.values())
        candidate = memory._copy()
        applied, rejected = candidate.apply_ops_atomically(ops)
        if rejected:
            return [], rejected

        after_active = sum(entry.status == "active" for entry in candidate.entries.values())
        if after_active >= before_active:
            return [], [
                {
                    "op": ops,
                    "reason": "compaction must reduce the number of active entries",
                }
            ]

        memory.entries = candidate.entries
        memory.narrative = candidate.narrative
        memory._counters = candidate._counters
        return applied, []

    def _normalize_ops(self, ops: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for op in ops:
            if isinstance(op, str) and op.strip().upper() == "NOOP":
                normalized.append({"op": "NOOP"})
                continue
            if not isinstance(op, dict):
                normalized.append(op)
                continue
            item = dict(op)
            if item.get("op") == "ADD":
                section = item.get("section")
                if isinstance(section, str):
                    item["section"] = self._section_key_by_prefix.get(
                        section.lower(),
                        section,
                    )
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

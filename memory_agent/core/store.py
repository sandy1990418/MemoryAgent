"""Transactional structured-memory store with auditable operations."""

from __future__ import annotations

from threading import RLock
from typing import Callable

from memory_agent.core.models import (
    MemoryEntry,
    MemoryPolicyRef,
)
from memory_agent.core.sections import CHAT_SECTIONS, SectionConfig


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


class Memory:
    """Structured living summary made of discrete, addressable entries.

    Entries are never rewritten wholesale: they can only be added, updated
    (text replaced, provenance extended), or superseded (marked inactive,
    never deleted). The only field allowed to freely degrade is `narrative`.
    """

    def __init__(
        self,
        sections: list[SectionConfig] | None = None,
        policy: MemoryPolicyRef | None = None,
    ) -> None:
        self.sections: list[SectionConfig] = sections if sections is not None else list(CHAT_SECTIONS)
        self.policy = policy
        self._section_by_key: dict[str, SectionConfig] = {s.key: s for s in self.sections}
        self.entries: dict[str, MemoryEntry] = {}
        self.narrative: str = ""
        self._counters: dict[str, int] = {s.key: 0 for s in self.sections}
        self._lock = RLock()
        self._revision = 0

    def _next_id(self, section_key: str) -> str:
        cfg = self._section_by_key[section_key]
        self._counters[section_key] += 1
        return f"{cfg.prefix}{self._counters[section_key]}"

    def apply_ops(self, ops: list[dict]) -> tuple[list[dict], list[dict]]:
        """Apply a batch of ops. Returns (applied, rejected). Never raises."""
        with self._lock:
            applied: list[dict] = []
            rejected: list[dict] = []
            for op in ops:
                try:
                    result = self._apply_one(op)
                except Exception as exc:  # defensive: malformed ops must never raise
                    rejected.append({"op": op, "reason": f"exception: {exc}"})
                    continue
                if result is True:
                    applied.append(op)
                else:
                    rejected.append({"op": op, "reason": result})
            if applied:
                self._revision += 1
            return applied, rejected

    def apply_ops_atomically(self, ops: list[dict]) -> tuple[list[dict], list[dict]]:
        """Apply a batch only when every op is valid.

        The lower-level apply_ops method is intentionally permissive and can
        partially apply a batch. Updater-generated batches are different: if
        any op is rejected, evicting the corresponding turns would risk losing
        facts that never reached memory. This method validates on a copy and
        commits only a fully accepted batch.
        """
        if not ops:
            return [], []
        with self._lock:
            candidate = self._copy()
            applied, rejected = candidate.apply_ops(ops)
            if rejected:
                return [], rejected
            self._replace_from(candidate)
            return applied, []

    def _copy(self) -> "Memory":
        clone = Memory(sections=list(self.sections), policy=self.policy)
        clone.entries = {
            entry_id: MemoryEntry(
                id=entry.id,
                section=entry.section,
                text=entry.text,
                provenance=list(entry.provenance),
                status=entry.status,
                note=entry.note,
            )
            for entry_id, entry in self.entries.items()
        }
        clone.narrative = self.narrative
        clone._counters = dict(self._counters)
        clone._revision = self._revision
        return clone

    def transaction_snapshot(self) -> tuple["Memory", int]:
        """Return an isolated copy and the live revision it was based on."""
        with self._lock:
            return self._copy(), self._revision

    def commit_trial(self, trial: "Memory", base_revision: int) -> None:
        """Atomically install a completed trial iff live memory did not change."""
        with self._lock:
            if self._revision != base_revision:
                raise RuntimeError("memory changed while update transaction was in progress")
            self._replace_from(trial)

    def _replace_from(self, source: "Memory") -> None:
        snapshot = source._copy()
        self.entries = snapshot.entries
        self.narrative = snapshot.narrative
        self._counters = snapshot._counters
        self._revision += 1

    def to_state(self) -> dict:
        """Serialize entries, narrative, and id counters for persistence."""
        entries = []
        for entry in self.entries.values():
            state: dict = {
                "id": entry.id,
                "section": entry.section,
                "text": entry.text,
                "provenance": list(entry.provenance),
                "status": entry.status,
                "note": entry.note,
            }
            entries.append(state)
        return {
            "entries": entries,
            "narrative": self.narrative,
            "counters": dict(self._counters),
        }

    def load_state(self, state: dict) -> None:
        """Restore a to_state() payload, replacing entries and counters."""
        entries: dict[str, MemoryEntry] = {}
        for raw in state.get("entries", []):
            section = raw.get("section")
            if section not in self._section_by_key:
                raise ValueError(f"unknown section in memory state: {section}")
            entry = MemoryEntry(
                id=str(raw["id"]),
                section=section,
                text=str(raw["text"]),
                provenance=list(raw.get("provenance", [])),
                status=raw.get("status", "active"),
                note=raw.get("note", ""),
            )
            entries[entry.id] = entry

        counters = {key: 0 for key in self._section_by_key}
        saved_counters = state.get("counters") or {}
        for key, count in saved_counters.items():
            if key in counters:
                counters[key] = int(count)
        # Guard against id collisions when counters are missing or stale.
        for entry in entries.values():
            prefix = self._section_by_key[entry.section].prefix
            suffix = entry.id[len(prefix):]
            if entry.id.startswith(prefix) and suffix.isdigit():
                counters[entry.section] = max(counters[entry.section], int(suffix))

        self.entries = entries
        self.narrative = str(state.get("narrative", ""))
        self._counters = counters
        self._revision += 1

    def _apply_one(self, op: dict):
        """Return True on success, or a string reason for rejection."""
        if not isinstance(op, dict):
            return "op is not a dict"

        kind = op.get("op")

        if kind == "NOOP":
            return True

        if kind == "ADD":
            section = op.get("section")
            text = op.get("text")
            provenance = op.get("provenance", [])
            if section not in self._section_by_key:
                return f"unknown section: {section}"
            if not isinstance(text, str) or not text:
                return "missing/invalid text"
            if not isinstance(provenance, list):
                return "invalid provenance"
            entry_id = self._next_id(section)
            self.entries[entry_id] = MemoryEntry(
                id=entry_id,
                section=section,
                text=text,
                provenance=list(provenance),
            )
            return True

        if kind == "UPDATE":
            entry_id = op.get("id")
            text = op.get("text")
            provenance = op.get("provenance", [])
            if entry_id not in self.entries:
                return self._unknown_entry_id_reason(entry_id)
            entry = self.entries[entry_id]
            if entry.status == "superseded":
                return f"cannot update superseded entry: {entry_id}"
            if not isinstance(text, str) or not text:
                return "missing/invalid text"
            if not isinstance(provenance, list):
                return "invalid provenance"
            entry.text = text
            entry.provenance = sorted(set(entry.provenance) | set(provenance))
            return True

        if kind == "SUPERSEDE":
            entry_id = op.get("id")
            reason = op.get("reason", "")
            if entry_id not in self.entries:
                return self._unknown_entry_id_reason(entry_id)
            entry = self.entries[entry_id]
            entry.status = "superseded"
            entry.note = reason
            return True

        return f"unknown op: {kind}"

    @staticmethod
    def _unknown_entry_id_reason(entry_id: object) -> str:
        if isinstance(entry_id, int) or (isinstance(entry_id, str) and entry_id.isdigit()):
            return (
                f"unknown memory entry id: {entry_id}; UPDATE/SUPERSEDE ids must be "
                "exact current memory entry ids like F1, U2, or G3, not turn_id values"
            )
        return f"unknown memory entry id: {entry_id}"

    def render(
        self,
        include_superseded: bool = False,
        max_tokens: int | None = None,
        token_estimator: Callable[[str], int] | None = None,
        entries: list[MemoryEntry] | None = None,
    ) -> str:
        lines: list[str] = []
        estimator = token_estimator or _default_token_estimator
        omitted = 0
        source_entries = list(self.entries.values()) if entries is None else list(entries)

        def would_fit(extra_lines: list[str]) -> bool:
            if max_tokens is None:
                return True
            text = "\n".join(lines + extra_lines).rstrip("\n")
            return estimator(text) <= max_tokens

        def append_entry_block(header: str, entry_lines: list[str]) -> None:
            nonlocal omitted
            section_started = False
            for line in entry_lines:
                extra = []
                if not section_started:
                    extra.append(header)
                extra.append(line)
                if would_fit(extra + [""]):
                    if not section_started:
                        lines.append(header)
                        section_started = True
                    lines.append(line)
                else:
                    omitted += 1
            if section_started:
                lines.append("")

        for cfg in self.sections:
            section_entries = [
                e for e in source_entries if e.section == cfg.key and e.status == "active"
            ]
            if not section_entries:
                continue
            entry_lines = [
                f"- [{e.id}] {e.text}"
                for e in section_entries
            ]
            append_entry_block(f"## {cfg.title}", entry_lines)

        if include_superseded:
            superseded_entries = [e for e in source_entries if e.status == "superseded"]
            if superseded_entries:
                entry_lines = []
                for e in superseded_entries:
                    note = f" - {e.note}" if e.note else ""
                    entry_lines.append(f"- [{e.id}] {e.text}{note}")
                append_entry_block("## Superseded", entry_lines)

        if self.narrative:
            narrative_lines = ["## Additional Narrative", self.narrative, ""]
            if would_fit(narrative_lines):
                lines.extend(narrative_lines)
            else:
                omitted += 1

        if omitted:
            notice = [
                "## Omitted",
                f"- {omitted} memory item(s) omitted because of the token budget.",
                "",
            ]
            if would_fit(notice):
                lines.extend(notice)

        return "\n".join(lines).rstrip("\n")

    def render_chronological(
        self,
        entries: list[MemoryEntry] | None = None,
        max_tokens: int | None = None,
        token_estimator: Callable[[str], int] | None = None,
        exclude_sections: frozenset[str] | set[str] | None = None,
    ) -> str:
        estimator = token_estimator or _default_token_estimator
        source_entries = list(self.entries.values()) if entries is None else list(entries)
        active_entries = [
            entry
            for entry in source_entries
            if entry.status == "active"
            and (exclude_sections is None or entry.section not in exclude_sections)
        ]
        if not active_entries:
            return ""

        sorted_entries = sorted(
            active_entries,
            key=lambda entry: (
                min(entry.provenance) if entry.provenance else float("inf"),
                entry.id,
            ),
        )
        lines = ["## Chronological Order (earliest mention first)"]
        omitted = 0

        for entry in sorted_entries:
            line = f"- [{entry.id}] {entry.text}"
            if max_tokens is None:
                lines.append(line)
                continue

            projected = "\n".join([*lines, line]).rstrip("\n")
            if estimator(projected) <= max_tokens:
                lines.append(line)
            else:
                omitted = len(sorted_entries) - (len(lines) - 1)
                break

        if omitted:
            lines.append(f"{omitted} memory item(s) omitted because of the token budget.")

        return "\n".join(lines).rstrip("\n")

    def set_narrative(self, text: str) -> None:
        self.narrative = text

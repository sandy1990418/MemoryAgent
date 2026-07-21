"""Structural recency/budget selection for answer-memory context."""

from __future__ import annotations

from typing import Callable

from memory_agent.core.models import MemoryEntry, SelectedMemory
from memory_agent.core.store import Memory
from memory_agent.policies.structured import StructuredMemoryPolicy


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


class MemorySelector:
    """Select memory by status/recency and hard token budget only.

    ``query`` is accepted at the API edge for callers that already provide it,
    but it never changes selection. Semantic ranking belongs to an explicit
    retrieval provider, not this bounded in-process fallback.  When multiple
    structural sections are present, selection interleaves their newest
    entries.  That keeps a large section (for example, ``facts``) from
    consuming the entire answer budget while still making the result
    deterministic and query-independent.
    """

    def __init__(
        self,
        token_estimator: Callable[[str], int] | None = None,
        policy: StructuredMemoryPolicy | None = None,
    ) -> None:
        self.token_estimator = token_estimator or _default_token_estimator
        self.policy = policy

    def select(
        self,
        memory: Memory,
        query: str = "",
        max_tokens: int | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        return [item.entry for item in self.select_with_scores(
            memory=memory,
            query=query,
            max_tokens=max_tokens,
            include_superseded=include_superseded,
        )]

    def select_for_answer(
        self,
        memory: Memory,
        query: str = "",
        budget: int | None = None,
    ) -> list[MemoryEntry]:
        return self.select(memory=memory, query=query, max_tokens=budget)

    def select_with_scores(
        self,
        memory: Memory,
        query: str = "",
        max_tokens: int | None = None,
        include_superseded: bool = False,
    ) -> list[SelectedMemory]:
        entries = [
            entry for entry in memory.entries.values()
            if include_superseded or entry.status == "active"
        ]
        entries = self._section_balanced_order(entries, memory)
        selected: list[SelectedMemory] = []
        selected_entries: list[MemoryEntry] = []
        for entry in entries:
            candidate_entries = [*selected_entries, entry]
            if max_tokens is not None:
                rendered = memory.render(entries=candidate_entries)
                if self.token_estimator(rendered) > max_tokens:
                    continue
            recency = min(max(entry.provenance), 1000) / 1000.0 if entry.provenance else 0.0
            selected.append(SelectedMemory(
                entry=entry,
                score=(1.0 if entry.status == "active" else 0.0) + recency,
                reasons=(
                    ("active",) if entry.status == "active" else ("superseded",)
                ) + (("recency",) if entry.provenance else ()),
            ))
            selected_entries.append(entry)
        return selected

    @staticmethod
    def _section_balanced_order(
        entries: list[MemoryEntry], memory: Memory
    ) -> list[MemoryEntry]:
        """Return newest-first entries while giving each section a turn.

        Section keys and entry provenance are the only ordering inputs.  The
        operation is intentionally independent of ``query`` and entry text so
        this fallback cannot grow benchmark- or language-specific semantics.
        Active entries are emitted before superseded history, preserving the
        historical ``include_superseded`` contract.
        """
        section_order = {
            section.key: index for index, section in enumerate(memory.sections)
        }
        # A custom Memory may contain a section not listed in its current
        # configuration. Keep this defensive fallback deterministic without
        # changing the normal chat section order.
        unknown_sections = sorted(
            {entry.section for entry in entries if entry.section not in section_order}
        )
        section_order.update(
            {section: len(section_order) + index for index, section in enumerate(unknown_sections)}
        )

        ordered: list[MemoryEntry] = []
        statuses = ("active", "superseded")
        for status in statuses:
            by_section: dict[str, list[MemoryEntry]] = {}
            for entry in entries:
                if entry.status != status:
                    continue
                by_section.setdefault(entry.section, []).append(entry)
            for section_entries in by_section.values():
                section_entries.sort(
                    key=lambda entry: (
                        -(max(entry.provenance) if entry.provenance else -1),
                        entry.id,
                    )
                )
            sections = sorted(by_section, key=lambda section: section_order[section])
            for rank in range(max((len(by_section[section]) for section in sections), default=0)):
                for section in sections:
                    section_entries = by_section[section]
                    if rank < len(section_entries):
                        ordered.append(section_entries[rank])
        return ordered

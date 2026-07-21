"""Evaluation-only heuristic quality signals for structured chat memory.

These diagnostics are intentionally outside :mod:`memory_agent`: they make
semantic judgments for analysis and must never affect production updates,
verification, retrieval, or compaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from memory_agent.core.store import Memory


@dataclass(frozen=True)
class QualityIndicator:
    count: int
    evidence: tuple[str, ...] = ()
    label: str = "heuristic"


@dataclass(frozen=True)
class MemoryQualityReport:
    canonical: QualityIndicator
    incomplete: QualityIndicator
    duplicate: QualityIndicator
    stale: QualityIndicator
    raw_request: QualityIndicator
    section_mismatch: QualityIndicator
    future_usefulness: QualityIndicator


_CANONICAL_PREFIXES = (
    "Ongoing state:", "Completed state:", "Goal:", "Constraint:",
    "Stable preference:", "User stated:",
)
_RAW_REQUEST_RE = re.compile(r"^(?:user\s+)?(?:asked|requested|wants?\s+me\s+to|please)\b", re.I)


def memory_quality_report(memory: Memory) -> MemoryQualityReport:
    """Return heuristic evaluation signals without changing runtime behavior."""
    active = [entry for entry in memory.entries.values() if entry.status == "active"]
    incomplete_ids = tuple(
        entry.id for entry in active
        if not entry.text.strip() or entry.text.rstrip().endswith(("…", "...", ":", ",", ";", "-"))
    )
    raw_ids = tuple(entry.id for entry in active if _RAW_REQUEST_RE.search(entry.text.strip()))
    canonical_ids = tuple(
        entry.id for entry in active
        if entry.text.startswith(_CANONICAL_PREFIXES) and entry.id not in incomplete_ids
    )
    seen: dict[tuple[str, str], str] = {}
    duplicate_ids: list[str] = []
    for entry in active:
        key = (entry.section, " ".join(entry.text.lower().split()))
        if key in seen:
            duplicate_ids.append(entry.id)
        else:
            seen[key] = entry.id
    stale = tuple(entry.id for entry in memory.entries.values() if entry.status == "superseded")
    mismatch = tuple(
        entry.id for entry in active
        if (entry.text.startswith("Goal:") and entry.section != "goal")
        or (entry.text.startswith(("Constraint:", "Stable preference:")) and entry.section != "preferences")
    )
    useful = tuple(entry.id for entry in active if entry.id not in incomplete_ids and entry.id not in raw_ids)

    def signal(ids: tuple[str, ...] | list[str]) -> QualityIndicator:
        evidence = tuple(ids)
        return QualityIndicator(len(evidence), evidence)

    return MemoryQualityReport(
        canonical=signal(canonical_ids), incomplete=signal(incomplete_ids),
        duplicate=signal(duplicate_ids), stale=signal(stale), raw_request=signal(raw_ids),
        section_mismatch=signal(mismatch),
        future_usefulness=signal(useful),
    )

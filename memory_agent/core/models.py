"""Framework-neutral structured-memory data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


MemoryEntryStatus = Literal["active", "superseded"]


class MemoryPolicyRef(Protocol):
    """Minimal policy identity retained with a memory snapshot."""

    name: str


@dataclass
class MemoryEntry:
    id: str
    section: str
    text: str
    provenance: list[int]
    status: MemoryEntryStatus = "active"
    note: str = ""


@dataclass(frozen=True)
class SelectedMemory:
    entry: MemoryEntry
    score: float
    reasons: tuple[str, ...]

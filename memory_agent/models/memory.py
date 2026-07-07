"""Data models used by structured memory and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MemoryEntryStatus = Literal["active", "superseded"]


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

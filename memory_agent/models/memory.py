"""Data models used by structured memory and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


MemoryEntryStatus = Literal["active", "superseded"]


@dataclass(frozen=True)
class SubjectIdentity:
    namespace: str
    entity: str
    attribute: str
    qualifier: str | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("subject identity confidence must be between 0 and 1")


@dataclass(frozen=True)
class MemoryValue:
    value: str
    unit: str | None = None


class SubjectNormalizer(Protocol):
    def normalize(self, text: str) -> tuple[SubjectIdentity, MemoryValue] | None: ...


@dataclass
class MemoryEntry:
    id: str
    section: str
    text: str
    provenance: list[int]
    status: MemoryEntryStatus = "active"
    note: str = ""
    subject_identity: SubjectIdentity | None = None
    value: MemoryValue | None = None


@dataclass(frozen=True)
class SelectedMemory:
    entry: MemoryEntry
    score: float
    reasons: tuple[str, ...]

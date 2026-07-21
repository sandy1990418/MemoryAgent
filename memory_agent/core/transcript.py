"""Framework-neutral transcript data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Turn:
    id: int
    role: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id < 1:
            raise ValueError("turn id must be a positive integer")
        if self.role not in {"user", "assistant"}:
            raise ValueError("turn role must be 'user' or 'assistant'")
        if not isinstance(self.content, str):
            raise TypeError("turn content must be a string")

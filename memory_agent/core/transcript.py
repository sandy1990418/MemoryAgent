"""Framework-neutral transcript data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Turn:
    id: int
    role: str
    content: str

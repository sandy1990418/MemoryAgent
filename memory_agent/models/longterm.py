"""Long-term memory data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LongTermHit:
    """One memory recalled from the long-term vector store."""

    text: str
    score: float | None = None
    metadata: dict | None = None

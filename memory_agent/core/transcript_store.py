"""Append-only transcript store."""

from __future__ import annotations

import json
from dataclasses import asdict

from memory_agent.core.transcript import Turn


class Transcript:
    """Append-only list of turns. No mutation or deletion methods."""

    def __init__(self) -> None:
        self._turns: list[Turn] = []
        self._next_id: int = 1

    def append(self, role: str, content: str) -> Turn:
        turn = Turn(id=self._next_id, role=role, content=content)
        self._turns.append(turn)
        self._next_id += 1
        return turn

    def get(self, start: int | None = None, end: int | None = None) -> list[Turn]:
        """Return turns with id in [start, end] (inclusive), both optional."""
        result = []
        for turn in self._turns:
            if start is not None and turn.id < start:
                continue
            if end is not None and turn.id > end:
                continue
            result.append(turn)
        return result

    def all(self) -> list[Turn]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def dump_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in self._turns], f, ensure_ascii=False, indent=2)

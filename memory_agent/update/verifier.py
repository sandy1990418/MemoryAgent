"""Structural verification for chat-memory update operations.

The updater model owns semantic retention decisions.  Verification is limited
to malformed operations, known sections/ids, provenance, and atomic safety;
there is intentionally no regex or dataset-specific semantic rubric here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn


@dataclass(frozen=True)
class MemoryUpdateVerification:
    passed: bool
    errors: list[str] = field(default_factory=list)


class MemoryUpdateVerifier:
    """Validate the structural contract of a completed update transaction."""

    def __init__(self, policy=None) -> None:
        # Kept as an adapter-compatible no-op argument. Policy selection is no
        # longer a verification concern; ``Memory`` supplies valid sections.
        self.policy = policy

    def verify(
        self,
        evicted_turns: list[Turn],
        applied_ops: list[dict],
        rejected_ops: list[dict],
        memory: Memory,
    ) -> MemoryUpdateVerification:
        errors: list[str] = []
        if rejected_ops:
            errors.append(f"Rejected ops exist: {rejected_ops}")

        allowed_sections = {section.key for section in memory.sections}
        allowed_turn_ids = {turn.id for turn in evicted_turns}
        for op in applied_ops:
            if not isinstance(op, dict):
                errors.append(f"Operation is not an object: {op!r}")
                continue
            kind = op.get("op")
            if kind == "NOOP":
                continue
            if kind == "ADD":
                section = op.get("section")
                if section not in allowed_sections:
                    errors.append(f"ADD uses unknown section: {section!r}")
                self._check_text(op, "ADD", errors)
                self._check_provenance(op, allowed_turn_ids, "ADD", errors)
            elif kind == "UPDATE":
                if op.get("id") not in memory.entries:
                    errors.append(f"UPDATE uses unknown id: {op.get('id')!r}")
                self._check_text(op, "UPDATE", errors)
                self._check_provenance(op, allowed_turn_ids, "UPDATE", errors)
            elif kind == "SUPERSEDE":
                if op.get("id") not in memory.entries:
                    errors.append(f"SUPERSEDE uses unknown id: {op.get('id')!r}")
            else:
                errors.append(f"Unknown operation: {kind!r}")

        return MemoryUpdateVerification(passed=not errors, errors=errors)

    @staticmethod
    def _check_text(op: dict, kind: str, errors: list[str]) -> None:
        text = op.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{kind} text must be a non-empty string")
        elif len(text) > 500:
            errors.append(f"{kind} text exceeds 500 characters")

    @staticmethod
    def _check_provenance(
        op: dict,
        allowed_turn_ids: set[int],
        kind: str,
        errors: list[str],
    ) -> None:
        provenance = op.get("provenance")
        if not isinstance(provenance, list) or not provenance:
            errors.append(f"{kind} provenance must be a non-empty list")
            return
        invalid = [turn_id for turn_id in provenance if turn_id not in allowed_turn_ids]
        if invalid:
            errors.append(f"{kind} provenance contains unknown turn ids: {invalid}")

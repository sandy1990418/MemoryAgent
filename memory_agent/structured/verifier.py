"""Lightweight semantic verifier for updater outputs.

Schema validation (JSON shape, known sections, real entry ids, provenance)
already happens inside `MemoryUpdater` / `Memory.apply_ops_atomically`. This
verifier adds a BEAM-motivated semantic invariant on top: when a batch of
evicted turns contains an explicit status-change cue ("never", "actually",
"changed my mind", ...), the update must leave that change recorded somewhere
— otherwise the turns must not be evicted, or the denial/correction is lost
for good.

It is deterministic (no LLM call) and is designed as defense-in-depth: the
deterministic extraction pass in `MemoryUpdater` normally guarantees the
invariant, so a failure here means the pipeline itself regressed.

One deliberate deviation from the naive design: a cue whose statement is
ALREADY covered by an active `status_changes` entry counts as satisfied. The
updater dedups repeated statements, so requiring a fresh op for an
already-recorded cue would fail verification on every retry, and the evicted
turns would never leave the context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory_agent.models.policy import MemoryPolicy
from memory_agent.models.transcript import Turn
from memory_agent.structured.memory import Memory
from memory_agent.structured.heuristics import status_change_cue_re
from memory_agent.structured.updater import MemoryUpdater


@dataclass(frozen=True)
class MemoryUpdateVerification:
    passed: bool
    errors: list[str] = field(default_factory=list)


class MemoryUpdateVerifier:
    """Checks BEAM-relevant update invariants before evicted turns are dropped."""

    def __init__(self, policy: MemoryPolicy | None = None) -> None:
        self.policy = policy

    def verify(
        self,
        evicted_turns: list[Turn],
        applied_ops: list[dict],
        rejected_ops: list[dict],
        memory: Memory,
    ) -> MemoryUpdateVerification:
        errors: list[str] = []
        policy = self.policy or memory.policy
        # Same policy-aware cue set as the deterministic extractor: a broader
        # verifier regex would fail turns the extractor never records, forever.
        cue_re = status_change_cue_re(policy)

        if rejected_ops:
            errors.append(f"Rejected ops exist: {rejected_ops}")

        for turn in evicted_turns:
            if turn.role != "user":
                continue
            if not cue_re.search(turn.content):
                continue
            if (
                policy is not None
                and policy.name == "practical"
                and MemoryUpdater._is_ordinary_non_durable_batch([turn], cue_re=cue_re)
            ):
                continue
            if not self._cue_is_recorded(turn, applied_ops, memory, cue_re):
                errors.append(
                    f"Status-change cue in turn {turn.id} but the update produced "
                    "neither a status_changes ADD nor a SUPERSEDE, and no active "
                    "status_changes entry covers it."
                )

        return MemoryUpdateVerification(passed=not errors, errors=errors)

    @staticmethod
    def _cue_is_recorded(turn: Turn, applied_ops: list[dict], memory: Memory, cue_re) -> bool:
        has_status_change_add = any(
            isinstance(op, dict)
            and op.get("op") == "ADD"
            and op.get("section") == "status_changes"
            for op in applied_ops
        )
        has_supersede = any(
            isinstance(op, dict) and op.get("op") == "SUPERSEDE" for op in applied_ops
        )
        if has_status_change_add or has_supersede:
            return True

        # Escape hatch against infinite retries: the statement may have been
        # recorded by an earlier batch (the updater dedups repeats).
        snippet = MemoryUpdater._extract_status_change_snippet(turn.content, cue_re=cue_re)
        if snippet is None:
            return False
        active_texts = {
            MemoryUpdater._text_key(entry.text)
            for entry in memory.entries.values()
            if entry.section == "status_changes" and entry.status == "active"
        }
        return MemoryUpdater._has_seen_text(f"User stated: {snippet}", active_texts)

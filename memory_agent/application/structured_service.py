"""Framework-neutral orchestration for the structured-memory runtime."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import StructuredMemoryPolicy
from memory_agent.update.compactor import MemoryCompactor
from memory_agent.update.updater import MemoryUpdater, UpdateFailed
from memory_agent.update.verifier import (
    MemoryUpdateVerification,
    MemoryUpdateVerifier,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuredUpdateResult:
    """Outcome of one verified, transactional structured-memory update."""

    applied_ops: list[dict] = field(default_factory=list)
    rejected_ops: list[dict] = field(default_factory=list)
    verification: MemoryUpdateVerification | None = None
    committed: bool = False
    failure_reason: str | None = None


class StructuredMemoryService:
    """Own update transactions independently of any agent framework.

    LangChain middleware and standalone chat facades both delegate here.  This
    keeps message adaptation at the edge and provides the application boundary
    that future event adapters can target after translating events into update
    inputs.
    """

    def __init__(
        self,
        *,
        memory: Memory,
        updater: MemoryUpdater,
        policy: StructuredMemoryPolicy | None = None,
        update_verifier: MemoryUpdateVerifier | None = None,
        compactor: MemoryCompactor | None = None,
        compact_min_active_entries: int = 30,
    ) -> None:
        self.memory = memory
        self.updater = updater
        self.policy = policy or memory.policy or updater.policy
        self._validate_component_policies()
        self.update_verifier = update_verifier or MemoryUpdateVerifier(
            policy=self.policy
        )
        self.compactor = compactor
        self.compact_min_active_entries = compact_min_active_entries
        self._last_compaction_failure_active: int | None = None
        self._compaction_retry_growth = 10
        self._compaction_checks: list[dict[str, object]] = []

    def _validate_component_policies(self) -> None:
        """Reject stacks whose components disagree about workload semantics."""
        component_policies = [self.memory.policy, self.updater.policy]
        conflicts = {
            candidate.name
            for candidate in component_policies
            if candidate is not None and candidate.name != self.policy.name
        }
        if conflicts:
            names = ", ".join(sorted(conflicts | {self.policy.name}))
            raise ValueError(f"structured memory components use conflicting policies: {names}")

    def update(self, turns: list[Turn]) -> StructuredUpdateResult:
        """Prepare, verify and atomically commit one update batch."""
        prepared = self.updater.prepare_update(self.memory, turns)
        if prepared.rejected_ops:
            return StructuredUpdateResult(
                applied_ops=prepared.applied_ops,
                rejected_ops=prepared.rejected_ops,
                failure_reason="rejected_ops",
            )

        verification = self.update_verifier.verify(
            evicted_turns=turns,
            applied_ops=prepared.applied_ops,
            rejected_ops=prepared.rejected_ops,
            memory=prepared.trial_memory,
        )
        if not verification.passed:
            return StructuredUpdateResult(
                applied_ops=prepared.applied_ops,
                verification=verification,
                failure_reason="verification_failed",
            )

        try:
            prepared.commit(self.memory)
        except RuntimeError:
            return StructuredUpdateResult(
                applied_ops=prepared.applied_ops,
                verification=verification,
                failure_reason="concurrent_change",
            )

        self.maybe_compact()
        return StructuredUpdateResult(
            applied_ops=prepared.applied_ops,
            verification=verification,
            committed=True,
        )

    def maybe_compact(self) -> None:
        """Consolidate candidates without making compaction update-critical."""
        active = sum(
            entry.status == "active" for entry in self.memory.entries.values()
        )
        total = len(self.memory.entries)
        diagnostic: dict[str, object] = {
            "compactor_enabled": self.compactor is not None,
            "memory_profile": self.policy.name,
            "active_entries_at_check": active,
            "total_entries_at_check": total,
            "threshold": self.compact_min_active_entries,
            "candidate_count": 0,
            "skip_reason": None,
            "attempted_calls": 0,
        }
        if self.compactor is None:
            diagnostic["skip_reason"] = "disabled_profile"
            self._compaction_checks.append(diagnostic)
            return
        candidates = self.compactor.detect_candidates(self.memory)
        if active <= self.compact_min_active_entries:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.reason == "progress-rollup"
            ]
            if not candidates:
                diagnostic["skip_reason"] = "below_scan_threshold"
                self.compactor.record_skip("below_scan_threshold")
                self._compaction_checks.append(diagnostic)
                return
        diagnostic["candidate_count"] = len(candidates)
        if not candidates:
            diagnostic["skip_reason"] = "no_candidates"
            self.compactor.record_skip("no_candidates")
            self._compaction_checks.append(diagnostic)
            return
        if (
            self._last_compaction_failure_active is not None
            and active
            < self._last_compaction_failure_active + self._compaction_retry_growth
        ):
            diagnostic["skip_reason"] = "candidate_circuit_breaker"
            self.compactor.record_skip("candidate_circuit_breaker")
            self._compaction_checks.append(diagnostic)
            return
        diagnostic["skip_reason"] = (
            "llm_candidate"
            if any(
                candidate.reason in {"semantic-overlap", "progress-rollup"}
                for candidate in candidates
            )
            else "deterministic_candidate"
        )
        attempted_before = self.compactor.metrics.attempted_calls
        try:
            applied, rejected = self.compactor.compact_candidates(
                self.memory, candidates
            )
        except UpdateFailed as exc:
            self._last_compaction_failure_active = active
            logger.warning("Memory compaction failed; continuing uncompacted: %s", exc)
            diagnostic["attempted_calls"] = (
                self.compactor.metrics.attempted_calls - attempted_before
            )
            self._compaction_checks.append(diagnostic)
            return
        if rejected:
            self._last_compaction_failure_active = active
            logger.warning(
                "Memory compaction ops rejected; continuing uncompacted: %s", rejected
            )
        elif applied:
            self._last_compaction_failure_active = None
            logger.info("Memory compaction applied %d ops", len(applied))
        diagnostic["attempted_calls"] = (
            self.compactor.metrics.attempted_calls - attempted_before
        )
        self._compaction_checks.append(diagnostic)

    def compaction_diagnostics(self) -> dict[str, object]:
        reasons = Counter(
            check["skip_reason"]
            for check in self._compaction_checks
            if check["skip_reason"]
        )
        return {
            "compactor_enabled": self.compactor is not None,
            "memory_profile": self.policy.name,
            "threshold": self.compact_min_active_entries,
            "checks": list(self._compaction_checks),
            "skip_reasons": dict(reasons),
            "candidate_count": sum(
                int(check["candidate_count"]) for check in self._compaction_checks
            ),
            "attempted_calls": sum(
                int(check["attempted_calls"]) for check in self._compaction_checks
            ),
        }

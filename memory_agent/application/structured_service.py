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
    diagnostics: dict[str, object] = field(default_factory=dict)


class StructuredMemoryService:
    """Own verified, atomic chat-memory updates and optional compaction."""

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
        self._last_update_diagnostics: dict[str, object] = {
            "planned_turn_ids": [],
            "planned_batch_turn_ids": [],
            "committed_turn_ids": [],
            "deferred_turn_ids": [],
            "dropped_turn_ids": [],
            "status": "idle",
        }

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
        """Prepare, verify and atomically commit one update transaction.

        ``MemoryUpdater.prepare_update`` stages every planned micro-batch on
        one isolated trial memory. Verification runs for each staged batch,
        and this method performs exactly one live-memory commit after all
        verifications pass.
        """
        try:
            prepared = self.updater.prepare_update(self.memory, turns)
        except UpdateFailed as exc:
            logger.warning("Chat memory updater failed; no state was committed: %s", exc)
            diagnostics = self.updater.update_diagnostics()
            self._last_update_diagnostics = diagnostics
            return StructuredUpdateResult(
                failure_reason=f"updater_failed: {exc}",
                diagnostics=diagnostics,
            )
        if prepared.rejected_ops:
            diagnostics = self.updater.update_diagnostics()
            self._last_update_diagnostics = diagnostics
            return StructuredUpdateResult(
                # Earlier batches may have changed only the isolated trial;
                # expose no applied operations when the outer transaction is
                # rejected and nothing reached live memory.
                applied_ops=[],
                rejected_ops=prepared.rejected_ops,
                failure_reason="rejected_ops",
                diagnostics=diagnostics,
            )

        batch_verifications: list[MemoryUpdateVerification] = []
        try:
            if prepared.batches:
                for batch in prepared.batches:
                    batch_verifications.append(
                        self.update_verifier.verify(
                            evicted_turns=list(batch.turns),
                            applied_ops=batch.applied_ops,
                            rejected_ops=batch.rejected_ops,
                            memory=prepared.trial_memory,
                        )
                    )
            else:
                # Preserve the historical empty-update verifier call and
                # result shape while still treating it as one transaction.
                batch_verifications.append(
                    self.update_verifier.verify(
                        evicted_turns=turns,
                        applied_ops=[],
                        rejected_ops=[],
                        memory=prepared.trial_memory,
                    )
                )
        except Exception as exc:  # verifier failures are transaction failures
            diagnostics = self.updater.update_diagnostics()
            diagnostics["status"] = "verification_failed"
            diagnostics["committed_turn_ids"] = []
            diagnostics["deferred_turn_ids"] = list(diagnostics.get("planned_turn_ids", []))
            self._last_update_diagnostics = diagnostics
            verification = MemoryUpdateVerification(
                passed=False,
                errors=[f"verifier exception: {exc}"],
            )
            return StructuredUpdateResult(
                applied_ops=[],
                verification=verification,
                failure_reason="verification_failed",
                diagnostics=diagnostics,
            )
        verification_errors = [
            error
            for index, result in enumerate(batch_verifications)
            for error in (
                result.errors
                if result.passed
                else [f"batch {index + 1}: {error}" for error in result.errors]
            )
        ]
        verification = MemoryUpdateVerification(
            passed=all(result.passed for result in batch_verifications) and not verification_errors,
            errors=verification_errors,
        )
        if not verification.passed:
            diagnostics = self.updater.update_diagnostics()
            diagnostics["status"] = "verification_failed"
            diagnostics["committed_turn_ids"] = []
            diagnostics["deferred_turn_ids"] = list(diagnostics.get("planned_turn_ids", []))
            self._last_update_diagnostics = diagnostics
            return StructuredUpdateResult(
                applied_ops=[],
                verification=verification,
                failure_reason="verification_failed",
                diagnostics=diagnostics,
            )

        try:
            prepared.commit(self.memory)
        except RuntimeError:
            diagnostics = self.updater.update_diagnostics()
            diagnostics["status"] = "concurrent_change"
            diagnostics["committed_turn_ids"] = []
            diagnostics["deferred_turn_ids"] = list(diagnostics.get("planned_turn_ids", []))
            self._last_update_diagnostics = diagnostics
            return StructuredUpdateResult(
                applied_ops=[],
                verification=verification,
                failure_reason="concurrent_change",
                diagnostics=diagnostics,
            )

        self.updater._mark_update_committed(prepared)
        diagnostics = self.updater.update_diagnostics()
        self._last_update_diagnostics = diagnostics
        self.maybe_compact()
        return StructuredUpdateResult(
            applied_ops=prepared.applied_ops,
            verification=verification,
            committed=True,
            diagnostics=diagnostics,
        )

    def update_diagnostics(self) -> dict[str, object]:
        """Return diagnostics for the latest update transaction."""
        return {
            key: list(value) if isinstance(value, list) else value
            for key, value in self._last_update_diagnostics.items()
        }

    def maybe_compact(self) -> None:
        """Consolidate candidates without making compaction update-critical."""
        active = sum(
            entry.status == "active" for entry in self.memory.entries.values()
        )
        total = len(self.memory.entries)
        diagnostic: dict[str, object] = {
            "compactor_enabled": self.compactor is not None,
            "policy": self.policy.name,
            "active_entries_at_check": active,
            "total_entries_at_check": total,
            "threshold": self.compact_min_active_entries,
            "candidate_count": 0,
            "skip_reason": None,
            "attempted_calls": 0,
        }
        if self.compactor is None:
            diagnostic["skip_reason"] = "disabled"
            self._compaction_checks.append(diagnostic)
            return
        candidates = self.compactor.detect_candidates(self.memory)
        if active <= self.compact_min_active_entries:
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
        diagnostic["skip_reason"] = "llm_candidate"
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
            "policy": self.policy.name,
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

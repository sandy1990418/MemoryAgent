"""Memory retention policies for product and evaluation workloads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryPolicy:
    """Controls how aggressively structured memory retains conversation data."""

    name: str
    section_preset: str
    allow_exact_values: bool
    allow_deterministic_subject_values: bool
    max_ops_per_batch: int | None
    default_noop: bool


PRACTICAL_POLICY = MemoryPolicy(
    name="practical",
    section_preset="practical",
    allow_exact_values=False,
    allow_deterministic_subject_values=False,
    max_ops_per_batch=1,
    default_noop=True,
)

AGENT_POLICY = MemoryPolicy(
    name="agent",
    section_preset="agent",
    allow_exact_values=False,
    allow_deterministic_subject_values=True,
    max_ops_per_batch=None,
    default_noop=False,
)

EVAL_POLICY = MemoryPolicy(
    name="eval",
    section_preset="eval",
    allow_exact_values=True,
    allow_deterministic_subject_values=True,
    max_ops_per_batch=5,
    default_noop=False,
)

MEMORY_POLICIES: dict[str, MemoryPolicy] = {
    "practical": PRACTICAL_POLICY,
    "agent": AGENT_POLICY,
    "eval": EVAL_POLICY,
    "beam": EVAL_POLICY,
}


def get_memory_policy(name: str | None) -> MemoryPolicy:
    """Return a named memory policy; ``None`` selects the product default."""
    normalized = (name or "practical").strip().lower()
    try:
        return MEMORY_POLICIES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(MEMORY_POLICIES))
        raise ValueError(f"memory profile must be one of: {choices}") from exc

"""Policies for the operation-based structured-memory runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StructuredMemoryPolicy:
    """Configuration contract for the operation-based structured runtime.

    This policy is deliberately distinct from ``EventMemoryPolicy`` in
    ``memory_agent.policies.event``. The former configures structured-memory
    extraction and operation handling; the latter classifies generic events at
    an ingestion boundary.
    """

    name: str
    section_preset: str
    allow_exact_values: bool
    allow_deterministic_subject_values: bool
    max_ops_per_batch: int | None
    retention_semantics: Literal["durable_chat", "agent", "evaluation"]
    subject_value_retention: Literal["personal_only", "exclude_counts", "all"]
    disallowed_sections: frozenset[str]


PRACTICAL_POLICY = StructuredMemoryPolicy(
    name="practical",
    section_preset="practical",
    allow_exact_values=False,
    allow_deterministic_subject_values=True,
    max_ops_per_batch=3,
    retention_semantics="durable_chat",
    subject_value_retention="exclude_counts",
    disallowed_sections=frozenset(
        {"exact_values", "timeline", "tool_facts", "progress"}
    ),
)

CHAT_POLICY = StructuredMemoryPolicy(
    name="chat",
    section_preset="chat",
    allow_exact_values=False,
    allow_deterministic_subject_values=True,
    max_ops_per_batch=3,
    retention_semantics="durable_chat",
    subject_value_retention="personal_only",
    disallowed_sections=frozenset(
        {"exact_values", "timeline", "tool_facts", "progress"}
    ),
)

AGENT_POLICY = StructuredMemoryPolicy(
    name="agent",
    section_preset="agent",
    allow_exact_values=False,
    allow_deterministic_subject_values=True,
    max_ops_per_batch=None,
    retention_semantics="agent",
    subject_value_retention="all",
    disallowed_sections=frozenset({"exact_values"}),
)

EVAL_POLICY = StructuredMemoryPolicy(
    name="eval",
    section_preset="eval",
    allow_exact_values=True,
    allow_deterministic_subject_values=True,
    max_ops_per_batch=5,
    retention_semantics="evaluation",
    subject_value_retention="all",
    disallowed_sections=frozenset(),
)

MEMORY_POLICIES: dict[str, StructuredMemoryPolicy] = {
    "chat": CHAT_POLICY,
    "practical": PRACTICAL_POLICY,
    "agent": AGENT_POLICY,
    "eval": EVAL_POLICY,
}


def is_chat_policy(policy: StructuredMemoryPolicy) -> bool:
    """Whether a structured policy uses chat retention semantics."""
    return policy.retention_semantics == "durable_chat"


def get_memory_policy(name: str | None) -> StructuredMemoryPolicy:
    """Return a named memory policy; ``None`` selects the product default."""
    normalized = (name or "chat").strip().lower()
    try:
        return MEMORY_POLICIES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(MEMORY_POLICIES))
        raise ValueError(f"memory profile must be one of: {choices}") from exc


def validate_policy_sections(policy: StructuredMemoryPolicy, sections: list) -> None:
    """Raise when a policy is paired with sections it is meant to exclude.

    Use `sections_for_preset(policy.section_preset)` to build a matching list.
    """
    keys = {section.key for section in sections}
    conflicts = sorted(keys & policy.disallowed_sections)
    if conflicts:
        raise ValueError(
            f"policy '{policy.name}' disallows sections {conflicts}; "
            "build sections with sections_for_preset(policy.section_preset)"
        )
    if not policy.allow_exact_values and "exact_values" in keys:
        raise ValueError(
            f"policy '{policy.name}' has allow_exact_values=False but the "
            "section list includes exact_values"
        )

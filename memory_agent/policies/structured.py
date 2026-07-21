"""The chat policy contract for the operation-based memory runtime.

Production memory has one workload and one retention contract. Evaluation
code may compare runners, but it must not select different runtime semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
@dataclass(frozen=True)
class StructuredMemoryPolicy:
    """Structural safety knobs for the sole chat retention contract."""

    name: str
    max_ops_per_batch: int | None
    disallowed_sections: frozenset[str]


CHAT_POLICY = StructuredMemoryPolicy(
    name="chat",
    max_ops_per_batch=3,
    disallowed_sections=frozenset({"exact_values", "timeline", "tool_facts"}),
)


def validate_policy_sections(policy: StructuredMemoryPolicy, sections: list) -> None:
    """Raise when the chat policy is paired with unsafe sections.

    This remains a structural guard for callers that construct ``Memory`` or
    ``MemoryUpdater`` directly.
    """
    keys = {section.key for section in sections}
    conflicts = sorted(keys & policy.disallowed_sections)
    if conflicts:
        raise ValueError(
            f"policy '{policy.name}' disallows sections {conflicts}"
        )

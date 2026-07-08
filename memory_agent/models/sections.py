"""Structured-memory section data models and defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SectionConfig:
    key: str
    prefix: str
    title: str
    description: str


DECISIONS = SectionConfig(
    key="decisions",
    prefix="D",
    title="Decisions",
    description="Confirmed decisions and conclusions",
)

PREFERENCES = SectionConfig(
    key="preferences",
    prefix="U",
    title="User Preferences",
    description=(
        "Durable user preferences and stable background that should guide "
        "future responses. Do not store temporary task requests here."
    ),
)

FACTS = SectionConfig(
    key="facts",
    prefix="F",
    title="Facts",
    description=(
        "Durable facts about the user, project, implemented features, stack, "
        "constraints, errors, and observed results."
    ),
)

EXACT_VALUES = SectionConfig(
    key="exact_values",
    prefix="V",
    title="Exact Values",
    description=(
        "Optional legacy exact-value inventory for numbers, dates, versions, "
        "identifiers, file paths, and URLs. Not included in default memory "
        "sections; prefer embedding important values in concise subject entries."
    ),
)

OPEN_QUESTIONS = SectionConfig(
    key="open_questions",
    prefix="Q",
    title="Open Questions",
    description=(
        "Explicit unresolved decisions, blockers, or follow-up questions that "
        "remain important after the current turn. Do not store ordinary "
        "one-off help requests."
    ),
)

GOAL = SectionConfig(
    key="goal",
    prefix="G",
    title="Task Goal",
    description="Task goal and acceptance criteria",
)

STATUS_CHANGES = SectionConfig(
    key="status_changes",
    prefix="C",
    title="Status Changes",
    description=(
        "Explicit corrections, contradictions, reversals, denials, or "
        "latest-vs-previous state changes. Capture both the changed subject and "
        "the current truth."
    ),
)

TIMELINE = SectionConfig(
    key="timeline",
    prefix="M",
    title="Timeline",
    description=(
        "Explicitly stated dated milestones or phase plans (e.g., 'Phase 2: "
        "Nov 16 - Dec 15'). Do not use this section to record which topic the "
        "user raised first - that ordering is derived automatically from each "
        "entry's turn provenance and needs no separate entry here."
    ),
)

PROGRESS = SectionConfig(
    key="progress",
    prefix="P",
    title="Progress",
    description=(
        "Chronological milestones, completed work, active sprint focus, and "
        "measured progress. Keep entries concise and tied to source turns."
    ),
)

TOOL_FACTS = SectionConfig(
    key="tool_facts",
    prefix="T",
    title="Tool Facts",
    description="Key facts learned from tool calls",
)

FAILED_ATTEMPTS = SectionConfig(
    key="failed_attempts",
    prefix="X",
    title="Failed Attempts",
    description="Approaches that were tried and failed",
)


CHAT_SECTIONS: list[SectionConfig] = [
    DECISIONS,
    PREFERENCES,
    FACTS,
    OPEN_QUESTIONS,
]

PRACTICAL_SECTIONS: list[SectionConfig] = [
    DECISIONS,
    PREFERENCES,
    STATUS_CHANGES,
    GOAL,
    PROGRESS,
    FACTS,
    OPEN_QUESTIONS,
    FAILED_ATTEMPTS,
]

AGENT_SECTIONS: list[SectionConfig] = [
    DECISIONS,
    PREFERENCES,
    STATUS_CHANGES,
    TIMELINE,
    GOAL,
    PROGRESS,
    FACTS,
    TOOL_FACTS,
    OPEN_QUESTIONS,
    FAILED_ATTEMPTS,
]

EVAL_SECTIONS: list[SectionConfig] = [
    *AGENT_SECTIONS,
    EXACT_VALUES,
]


def sections_for_preset(name: str) -> list[SectionConfig]:
    """Return a fresh section list for a policy section preset."""
    presets = {
        "practical": PRACTICAL_SECTIONS,
        "agent": AGENT_SECTIONS,
        "eval": EVAL_SECTIONS,
    }
    try:
        return list(presets[name])
    except KeyError as exc:
        choices = ", ".join(sorted(presets))
        raise ValueError(f"section preset must be one of: {choices}") from exc

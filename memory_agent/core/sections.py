"""Structured-memory schema and section defaults."""

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

PROGRESS = SectionConfig(
    key="progress",
    prefix="P",
    title="Progress",
    description=(
        "Chronological milestones, completed work, active sprint focus, and "
        "measured progress. Keep entries concise and tied to source turns."
    ),
)

FAILED_ATTEMPTS = SectionConfig(
    key="failed_attempts",
    prefix="X",
    title="Failed Attempts",
    description="Approaches that were tried and failed",
)


# Product chat uses one bounded
# chronological rollup. Related progress entries are compacted into a new
# canonical summary rather than retained as an ever-growing event log.
CHAT_SECTIONS: list[SectionConfig] = [
    DECISIONS,
    PREFERENCES,
    STATUS_CHANGES,
    GOAL,
    FACTS,
    PROGRESS,
    OPEN_QUESTIONS,
    FAILED_ATTEMPTS,
]

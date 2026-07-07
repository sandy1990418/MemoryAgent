"""Structured-memory section data models and defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SectionConfig:
    key: str
    prefix: str
    title: str
    description: str


CHAT_SECTIONS: list[SectionConfig] = [
    SectionConfig(
        key="decisions",
        prefix="D",
        title="Decisions",
        description="Confirmed decisions and conclusions",
    ),
    SectionConfig(
        key="preferences",
        prefix="U",
        title="User Preferences",
        description=(
            "Durable user preferences and stable background that should guide "
            "future responses. Do not store temporary task requests here."
        ),
    ),
    SectionConfig(
        key="facts",
        prefix="F",
        title="Facts",
        description=(
            "Durable facts about the user, project, implemented features, stack, "
            "constraints, errors, and observed results."
        ),
    ),
    SectionConfig(
        key="open_questions",
        prefix="Q",
        title="Open Questions",
        description=(
            "Explicit unresolved decisions, blockers, or follow-up questions that "
            "remain important after the current turn. Do not store ordinary "
            "one-off help requests."
        ),
    ),
    SectionConfig(
        key="exact_values",
        prefix="V",
        title="Exact Values",
        description=(
            "Exact values that must be preserved verbatim: numbers, quantities, "
            "dates, versions, identifiers, file paths, URLs. Never paraphrase, "
            "round, or reword."
        ),
    ),
]

AGENT_SECTIONS: list[SectionConfig] = [
    *CHAT_SECTIONS,
    SectionConfig(
        key="goal",
        prefix="G",
        title="Task Goal",
        description="Task goal and acceptance criteria",
    ),
    SectionConfig(
        key="progress",
        prefix="P",
        title="Progress",
        description=(
            "Chronological milestones, completed work, active sprint focus, and "
            "measured progress. Keep entries concise and tied to source turns."
        ),
    ),
    SectionConfig(
        key="timeline",
        prefix="M",
        title="Timeline",
        description=(
            "Ordered project milestones, dates, phases, and event sequences that "
            "may be needed to answer chronology questions."
        ),
    ),
    SectionConfig(
        key="status_changes",
        prefix="C",
        title="Status Changes",
        description=(
            "Explicit corrections, contradictions, reversals, or latest-vs-previous "
            "state changes. Capture both the changed subject and the current truth."
        ),
    ),
    SectionConfig(
        key="tool_facts",
        prefix="T",
        title="Tool Facts",
        description="Key facts learned from tool calls",
    ),
    SectionConfig(
        key="failed_attempts",
        prefix="X",
        title="Failed Attempts",
        description="Approaches that were tried and failed",
    ),
]

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
        description="User preferences and relevant background",
    ),
    SectionConfig(
        key="facts",
        prefix="F",
        title="Facts",
        description="Facts established during the conversation",
    ),
    SectionConfig(
        key="open_questions",
        prefix="Q",
        title="Open Questions",
        description="Open questions that still need resolution",
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
        description="Plan and execution progress",
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

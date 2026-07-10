"""Entries stored by the framework-neutral memory core."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .memory_scope import MemoryScope


class MemoryType(str, Enum):
    USER_FACT = "user_fact"
    USER_PREFERENCE = "user_preference"
    USER_CONSTRAINT = "user_constraint"
    INSTRUCTION = "instruction"
    DECISION = "decision"
    GOAL = "goal"
    PLAN = "plan"
    TASK_STATE = "task_state"
    PROGRESS = "progress"
    STATUS_CHANGE = "status_change"
    TIMELINE_EVENT = "timeline_event"
    OBSERVATION = "observation"
    TOOL_RESULT = "tool_result"
    FAILED_ATTEMPT = "failed_attempt"
    SUCCESSFUL_STRATEGY = "successful_strategy"
    PROCEDURE = "procedure"
    ARTIFACT_REFERENCE = "artifact_reference"
    ENVIRONMENT_FACT = "environment_fact"
    UNRESOLVED_ISSUE = "unresolved_issue"
    LEARNED_KNOWLEDGE = "learned_knowledge"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class ProvenanceRef:
    event_id: str
    source_type: str | None = None


@dataclass
class MemoryEntry:
    memory_id: str
    memory_type: MemoryType
    scope: MemoryScope
    subject: str
    content: str
    status: MemoryStatus = MemoryStatus.ACTIVE
    provenance: list[ProvenanceRef] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    importance: float = 0.5
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.importance <= 1 or not 0 <= self.confidence <= 1:
            raise ValueError("importance and confidence must be between 0 and 1")

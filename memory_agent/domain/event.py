"""Generic events accepted by the memory pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class EventSourceType(str, Enum):
    CHAT_MESSAGE = "chat_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_DECISION = "agent_decision"
    TASK_STATUS = "task_status"
    USER_FEEDBACK = "user_feedback"
    OBSERVATION = "observation"
    ARTIFACT = "artifact"
    SYSTEM_EVENT = "system_event"


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    source_type: EventSourceType
    actor: str
    content: str
    timestamp: datetime | None = None
    session_id: str | None = None
    task_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

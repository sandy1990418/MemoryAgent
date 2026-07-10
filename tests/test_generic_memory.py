from dataclasses import FrozenInstanceError

import pytest

from memory_agent.application import MemoryService
from memory_agent.domain import EventSourceType, MemoryEvent, MemoryScope, MemoryType
from memory_agent.evaluation.agent import CriticalStateEvaluator
from memory_agent.profiles.agent import AgentEventAdapter, AgentMemoryPolicy
from memory_agent.profiles.chat import ChatEventAdapter, ChatMemoryPolicy
from memory_agent.profiles.policy import MemoryPolicy


def test_generic_event_is_immutable_and_not_chat_bound():
    event = MemoryEvent("e1", EventSourceType.TOOL_RESULT, "deploy", "failed", task_id="t1", metadata={"exit_code": 1})
    assert event.task_id == "t1"
    with pytest.raises(FrozenInstanceError):
        event.content = "changed"


def test_chat_adapter_is_the_only_place_that_needs_chat_roles():
    events = ChatEventAdapter().adapt([{"id": "u1", "role": "user", "content": "I prefer concise answers."}], session_id="s1")
    assert events[0].source_type == EventSourceType.CHAT_MESSAGE
    assert events[0].metadata["chat_role"] == "user"
    assert isinstance(ChatMemoryPolicy(), MemoryPolicy)


def test_agent_memory_keeps_failure_cause_but_drops_raw_tool_output():
    raw = "RAW_LOG_SENTINEL " * 500
    records = [
        {"source_type": "tool_result", "actor": "deploy", "content": raw},
        {"source_type": "task_status", "actor": "agent", "content": "Deployment failed because staging lacks API access.", "metadata": {"subject": "deployment", "critical": True}},
    ]
    events = AgentEventAdapter().adapt(records, task_id="t1")
    service = MemoryService(AgentMemoryPolicy())
    service.ingest_events(events)
    context = service.build_context(max_tokens=100)
    assert "staging lacks API access" in context
    assert "RAW_LOG_SENTINEL" not in context
    assert len(context) // 4 <= 100


def test_agent_memory_retains_unfinished_task_state_and_evaluation_extension():
    records = [
        {"event_id": "start", "source_type": "task_status", "content": "API client migration is in progress.", "metadata": {"subject": "migration"}},
        {"event_id": "block", "source_type": "task_status", "content": "Gemini Responses API adapter is blocked.", "metadata": {"subject": "gemini", "critical": True}},
    ]
    events = AgentEventAdapter().adapt(records, task_id="t2")
    service = MemoryService(AgentMemoryPolicy())
    service.ingest_events(events)
    context = service.build_context(max_tokens=100)
    assert "migration is in progress" in context
    assert "adapter is blocked" in context
    result = CriticalStateEvaluator().evaluate(events, list(service.entries.values()))
    assert result.critical_state_recall == 1.0


def test_profile_priorities_are_workload_specific():
    chat = ChatMemoryPolicy()
    agent = AgentMemoryPolicy()
    assert chat.name == "chat" and agent.name == "agent"
    assert MemoryScope.TASK.value == "task"
    assert MemoryType.FAILED_ATTEMPT.value == "failed_attempt"

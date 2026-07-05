import json
import re

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from memory_agent.langchain_middleware import StructuredMemoryMiddleware
from memory_agent.memory import Memory
from memory_agent.sections import CHAT_SECTIONS
from memory_agent.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


class DummyModel:
    """Stand-in for a BaseChatModel; ModelRequest does no runtime type check."""


def count_messages(messages):
    return len(messages)


def make_updater(script):
    return MemoryUpdater(llm=ScriptedLLM(script), sections=CHAT_SECTIONS)


def add_all_evicted_script(system, messages):
    """Extract every turn_id embedded in the prompt and ADD them as one entry."""
    turn_ids = [int(m) for m in re.findall(r'"turn_id":\s*(\d+)', system)]
    return json.dumps(
        [{"op": "ADD", "section": "facts", "text": "evicted context", "provenance": turn_ids}]
    )


def make_middleware(
    max_tokens,
    evict_fraction=0.5,
    script=add_all_evicted_script,
    memory=None,
    max_memory_tokens=None,
):
    updater = make_updater(script)
    return StructuredMemoryMiddleware(
        memory=memory if memory is not None else Memory(sections=CHAT_SECTIONS),
        updater=updater,
        max_tokens=max_tokens,
        evict_fraction=evict_fraction,
        max_memory_tokens=max_memory_tokens,
        token_counter=count_messages,
    )


def linear_messages(n):
    """n alternating human/ai messages with plain text content."""
    messages = []
    for i in range(n):
        if i % 2 == 0:
            messages.append(HumanMessage(content=f"human message {i}"))
        else:
            messages.append(AIMessage(content=f"ai message {i}"))
    return messages


def test_under_threshold_returns_none_but_mirrors_transcript():
    middleware = make_middleware(max_tokens=6)
    messages = linear_messages(3)

    result = middleware.before_model({"messages": messages}, None)

    assert result is None
    assert len(middleware.transcript) == 3


def test_over_threshold_evicts_and_populates_memory_with_correct_provenance():
    middleware = make_middleware(max_tokens=6, evict_fraction=0.5)
    messages = linear_messages(8)

    result = middleware.before_model({"messages": messages}, None)

    assert result is not None
    result_messages = result["messages"]
    assert isinstance(result_messages[0], RemoveMessage)
    assert result_messages[0].id == REMOVE_ALL_MESSAGES

    preserved = result_messages[1:]
    cutoff = len(messages) - len(preserved)
    assert cutoff > 0
    expected_preserved = messages[cutoff:]
    assert [m.id for m in preserved] == [m.id for m in expected_preserved]
    assert not isinstance(preserved[0], ToolMessage)

    evicted = messages[:cutoff]
    expected_turn_ids = sorted(
        middleware._turn_id_by_message_id[m.id] for m in evicted
    )

    facts_entries = [e for e in middleware.memory.entries.values() if e.section == "facts"]
    assert len(facts_entries) == 1
    assert sorted(facts_entries[0].provenance) == expected_turn_ids


def test_safe_cutoff_keeps_tool_call_pairs_together():
    middleware = make_middleware(max_tokens=6, evict_fraction=0.5)

    ai_with_tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "weather", "args": {"city": "Taipei"}, "id": "call1"}],
    )
    tool_response = ToolMessage(content="sunny", tool_call_id="call1", name="weather")

    messages = [
        HumanMessage(content="human 0"),
        AIMessage(content="ai 1"),
        HumanMessage(content="human 2"),
        ai_with_tool_call,
        tool_response,
        HumanMessage(content="human 5"),
        AIMessage(content="ai 6"),
    ]

    result = middleware.before_model({"messages": messages}, None)

    assert result is not None
    result_messages = result["messages"]
    preserved = result_messages[1:]
    cutoff = len(messages) - len(preserved)

    ai_index = messages.index(ai_with_tool_call)
    tool_index = messages.index(tool_response)
    ai_evicted = ai_index < cutoff
    tool_evicted = tool_index < cutoff
    assert ai_evicted == tool_evicted

    assert not isinstance(preserved[0], ToolMessage)


def test_updater_transport_error_returns_none_and_keeps_messages():
    def raising_script(system, messages):
        raise RuntimeError("network down")

    middleware = make_middleware(max_tokens=6, script=raising_script)
    messages = linear_messages(8)

    result = middleware.before_model({"messages": messages}, None)

    assert result is None
    assert middleware.memory.entries == {}
    assert len(middleware.transcript) == 8


def test_all_ops_rejected_returns_none_and_memory_unchanged():
    def rejecting_script(system, messages):
        return '[{"op": "UPDATE", "id": "D999", "text": "x", "provenance": [1]}]'

    middleware = make_middleware(max_tokens=6, script=rejecting_script)
    messages = linear_messages(8)

    result = middleware.before_model({"messages": messages}, None)

    assert result is None
    assert middleware.memory.entries == {}
    assert len(middleware.transcript) == 8


def test_mirroring_is_idempotent_across_calls():
    middleware = make_middleware(max_tokens=6)
    messages = linear_messages(3)

    first = middleware.before_model({"messages": messages}, None)
    second = middleware.before_model({"messages": messages}, None)

    assert first is None
    assert second is None
    assert len(middleware.transcript) == 3


def test_wrap_model_call_injects_rendered_memory_into_system_prompt():
    memory = Memory(sections=CHAT_SECTIONS)
    applied, rejected = memory.apply_ops_atomically(
        [
            {
                "op": "ADD",
                "section": "decisions",
                "text": "use in-memory storage for the cache layer",
                "provenance": [1],
            }
        ]
    )
    assert rejected == []

    middleware = make_middleware(max_tokens=6, memory=memory, max_memory_tokens=200)

    request = ModelRequest(model=DummyModel(), messages=[], system_prompt="Base prompt")

    captured = {}

    def handler(req):
        captured["request"] = req
        return req

    middleware.wrap_model_call(request, handler)

    final_prompt = captured["request"].system_prompt
    assert "Base prompt" in final_prompt
    assert "# Conversation Memory" in final_prompt
    assert "use in-memory storage for the cache layer" in final_prompt

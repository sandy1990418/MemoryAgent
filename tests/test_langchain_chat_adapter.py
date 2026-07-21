import ast
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from memory_agent.adapters.langchain.chat_memory import LangChainChatAdapter
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _updater(response):
    return MemoryUpdater(
        llm=ScriptedLLM(response),
        sections=CHAT_SECTIONS,
    )


def _adapter(*, response, **kwargs):
    return LangChainChatAdapter(
        Memory(CHAT_SECTIONS),
        _updater(response),
        max_tokens=kwargs.pop("max_tokens", 4),
        keep_messages=kwargs.pop("keep_messages", 2),
        token_counter=kwargs.pop("token_counter", lambda messages: len(messages)),
        **kwargs,
    )


def test_chat_adapter_has_no_agent_or_langgraph_imports():
    for path in (
        Path("memory_agent/adapters/langchain/chat.py"),
        Path("memory_agent/adapters/langchain/chat_memory.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            name.startswith(("langchain.agents", "langgraph")) for name in imports
        )


def test_core_imports_without_optional_frameworks_in_fresh_process():
    script = """
import builtins

real_import = builtins.__import__
blocked = ("langchain", "langgraph", "mem0", "openai")
def guarded(name, *args, **kwargs):
    if name == blocked[0] or name.startswith(blocked[0] + "."):
        raise ModuleNotFoundError(name)
    if name == blocked[1] or name.startswith(blocked[1] + "."):
        raise ModuleNotFoundError(name)
    if name == blocked[2] or name.startswith(blocked[2] + "."):
        raise ModuleNotFoundError(name)
    if name == blocked[3] or name.startswith(blocked[3] + "."):
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from memory_agent.core import Memory, Turn, Transcript
from memory_agent.application.structured_service import StructuredMemoryService
assert Memory and Turn and Transcript and StructuredMemoryService
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_failed_update_retains_exact_pending_messages_for_retry():
    calls = []
    failing = {"enabled": True}

    def fail_once(system, messages):
        calls.append(messages)
        if failing["enabled"]:
            raise RuntimeError("transport down")
        return json.dumps(
            [{"op": "ADD", "section": "facts", "text": "saved", "provenance": [1]}]
        )

    adapter = _adapter(response=fail_once, max_tokens=2)
    messages = [HumanMessage(content="old"), AIMessage(content="answer"), HumanMessage(content="new")]
    first = adapter.before_model(messages)
    first_ids = [message.id for message in first]
    assert [message.id for message in adapter.messages] == first_ids
    assert adapter.memory.entries == {}
    diagnostics = adapter.eviction_diagnostics()
    assert diagnostics["planned_turn_ids"] == [1]
    assert diagnostics["planned_batch_turn_ids"] == [[1]]
    assert diagnostics["committed_turn_ids"] == []
    assert diagnostics["deferred_turn_ids"] == [1]
    assert diagnostics["dropped_turn_ids"] == []
    assert diagnostics["status"] == "deferred"
    assert diagnostics["eviction_records"][-1]["deferred_turn_ids"] == [1]

    failing["enabled"] = False
    second = adapter.before_model(messages)
    assert len(second) == 2
    assert adapter.memory.entries
    assert [message.id for message in second] == [message.id for message in messages[-2:]]
    diagnostics = adapter.eviction_diagnostics()
    assert diagnostics["planned_turn_ids"] == [1]
    assert diagnostics["planned_batch_turn_ids"] == [[1]]
    assert diagnostics["committed_turn_ids"] == [1]
    assert diagnostics["deferred_turn_ids"] == []
    assert diagnostics["dropped_turn_ids"] == []
    assert diagnostics["status"] == "committed"
    assert diagnostics["eviction_records"][-1]["committed_turn_ids"] == [1]


def test_tool_messages_are_not_durable_but_pair_cutoff_is_safe():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "call-1"}],
    )
    tool = ToolMessage(content="transient result", tool_call_id="call-1", name="lookup")
    messages = [HumanMessage(content="question"), ai, tool, HumanMessage(content="follow-up")]
    adapter = _adapter(
        response=lambda *_: json.dumps(
            [{"op": "ADD", "section": "facts", "text": "saved", "provenance": [1]}]
        ),
        max_tokens=3,
        keep_messages=1,
    )

    # A cutoff inside the AI/tool group moves before the AI call.
    assert adapter._find_safe_cutoff_point(messages, 2) == 1
    retained = adapter.before_model(messages)
    assert retained[0].type != "tool"
    assert not any(turn.role == "tool" for turn in adapter.transcript.all())
    diagnostics = adapter.eviction_diagnostics()
    assert diagnostics["dropped_turn_ids"] == []
    assert diagnostics["deferred_turn_ids"] == []


def test_ainvoke_uses_async_model_and_retains_reply():
    class FakeAsyncChatModel:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, messages, **kwargs):
            self.calls.append(messages)
            return AIMessage(content="async reply")

    model = FakeAsyncChatModel()
    adapter = _adapter(response=lambda *_: "[]", chat_model=model, max_tokens=100)

    response = asyncio.run(adapter.ainvoke("hello"))

    assert response.content == "async reply"
    assert len(model.calls) == 1
    assert adapter.messages[-1].content == "async reply"


def test_invoke_injects_memory_at_answer_time_and_keeps_reply():
    class FakeChatModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages, **kwargs):
            self.calls.append(messages)
            return AIMessage(content="reply")

    adapter = _adapter(
        response=lambda *_: "[]",
        chat_model=FakeChatModel(),
        base_system_prompt="Be concise.",
        max_memory_tokens=200,
    )
    adapter.memory.apply_ops_atomically(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "Use SQLite.",
                "provenance": [1],
            }
        ]
    )
    response = adapter.invoke("Where should data live?")

    assert response.content == "reply"
    assert isinstance(adapter.chat_model.calls[0][0], SystemMessage)
    assert "Be concise." in adapter.chat_model.calls[0][0].content
    assert "Use SQLite." in adapter.chat_model.calls[0][0].content
    assert adapter.messages[-1].content == "reply"

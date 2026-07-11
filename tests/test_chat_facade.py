import importlib
import ast
import json
import sys
from pathlib import Path

from memory_agent.models.config import ProductMemoryConfig
from memory_agent.models.transcript import Turn
from tests.fakes import ScriptedLLM


def _clear_forbidden_modules() -> None:
    for name in list(sys.modules):
        if (
            name.startswith("memory_agent.agents")
            or name.startswith("scripts.")
            or name.startswith("memory_agent.clients.mem0")
        ):
            sys.modules.pop(name)


def test_chat_facade_updates_practical_memory_without_agent_imports():
    _clear_forbidden_modules()
    from memory_agent.chat import build_chat_memory

    chat = build_chat_memory(
        ScriptedLLM(
            lambda system, messages: json.dumps(
                [
                    {
                        "op": "ADD",
                        "section": "preferences",
                        "text": "User prefers concise replies.",
                        "provenance": [1],
                    }
                ]
            )
        ),
        compact=False,
    )

    applied, rejected = chat.update(
        [Turn(id=1, role="user", content="Remember I prefer concise replies.")]
    )

    assert rejected == []
    assert applied
    assert "User prefers concise replies." in chat.render()
    assert not any(name.startswith("memory_agent.agents") for name in sys.modules)
    assert not any(name.startswith("scripts.") for name in sys.modules)
    assert not any(name.startswith("memory_agent.clients.mem0") for name in sys.modules)


def test_chat_module_import_has_no_forbidden_reverse_dependencies():
    _clear_forbidden_modules()

    module = importlib.import_module("memory_agent.chat")

    assert hasattr(module, "build_chat_memory")
    assert not any(name.startswith("memory_agent.clients.mem0") for name in sys.modules)
    assert not any(name.startswith("memory_agent.agents") for name in sys.modules)


def test_chat_source_has_no_forbidden_imports():
    source = Path("memory_agent/chat.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    forbidden = (
        "memory_agent.agents",
        "memory_agent.clients.mem0",
        "scripts",
    )
    assert not any(name.startswith(forbidden) for name in imported)


def test_chat_compacts_only_above_configured_threshold():
    calls = []

    def responder(system, messages):
        calls.append(system)
        return json.dumps(
            [
                {
                    "op": "ADD",
                    "section": "preferences",
                    "text": "User prefers concise replies.",
                    "provenance": [1],
                }
            ]
        )

    from memory_agent.chat import build_chat_memory

    chat = build_chat_memory(
        ScriptedLLM(responder),
        config=ProductMemoryConfig(compaction_threshold=30),
    )
    chat.update([Turn(id=1, role="user", content="I prefer concise replies")])

    assert len(calls) == 1


def test_chat_records_token_usage_per_role():
    from memory_agent.chat import build_chat_memory

    chat = build_chat_memory(
        ScriptedLLM(
            lambda system, messages: json.dumps(
                [
                    {
                        "op": "ADD",
                        "section": "preferences",
                        "text": "User prefers concise replies.",
                        "provenance": [1],
                    }
                ]
            )
        ),
        compact=False,
    )

    chat.update([Turn(id=1, role="user", content="Remember I prefer concise replies.")])

    usage = chat.token_usage()
    assert set(usage) == {"updater", "compactor"}
    assert usage["updater"]["calls"] >= 1
    assert usage["updater"]["input_tokens"] > 0
    assert usage["updater"]["output_tokens"] > 0
    assert usage["compactor"]["calls"] == 0


def test_chat_compacts_after_crossing_configured_threshold():
    calls = []

    def responder(system, messages):
        calls.append(system)
        if len(calls) == 1:
            return json.dumps(
                [
                    {
                        "op": "ADD",
                        "section": "preferences",
                        "text": "User prefers concise replies.",
                        "provenance": [1],
                    }
                ]
            )
        return '[{"op": "NOOP"}]'

    from memory_agent.chat import build_chat_memory

    chat = build_chat_memory(
        ScriptedLLM(responder),
        config=ProductMemoryConfig(compaction_threshold=2),
    )
    applied, rejected = chat.memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "Project uses Flask.",
                "provenance": [2],
            },
            {
                "op": "ADD",
                "section": "facts",
                "text": "Project uses SQLite.",
                "provenance": [3],
            },
        ]
    )
    assert rejected == []
    assert len(applied) == 2

    chat.update([Turn(id=1, role="user", content="I prefer concise replies")])

    assert len(calls) == 1
    assert chat.compactor is not None
    assert chat.compactor.metrics.attempted_calls == 0

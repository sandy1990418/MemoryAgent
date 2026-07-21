import importlib
import ast
import json
import sys
from pathlib import Path

import pytest

from memory_agent.core.transcript import Turn
from memory_agent.models.config import ProductMemoryConfig
from tests.fakes import ScriptedLLM


def _clear_forbidden_modules() -> None:
    for name in list(sys.modules):
        if (
            name.startswith("memory_agent.agents")
            or name.startswith("scripts.")
            or name.startswith("memory_agent.clients.mem0")
        ):
            sys.modules.pop(name)


def test_turn_accepts_only_chat_roles():
    with pytest.raises(ValueError, match="user.*assistant"):
        Turn(1, "tool", "transient observation")


def test_chat_facade_updates_memory_without_optional_imports():
    _clear_forbidden_modules()
    from memory_agent.application.chat import build_chat_memory

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


def test_application_chat_module_import_has_no_forbidden_reverse_dependencies():
    _clear_forbidden_modules()

    module = importlib.import_module("memory_agent.application.chat")

    assert hasattr(module, "build_chat_memory")
    assert not any(name.startswith("memory_agent.clients.mem0") for name in sys.modules)
    assert not any(name.startswith("memory_agent.agents") for name in sys.modules)


def test_chat_source_has_no_forbidden_imports():
    source = Path("memory_agent/application/chat.py").read_text(encoding="utf-8")
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
        "evaluation",
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

    from memory_agent.application.chat import build_chat_memory

    chat = build_chat_memory(
        ScriptedLLM(responder),
        config=ProductMemoryConfig(compaction_threshold=30),
    )
    chat.update([Turn(id=1, role="user", content="I prefer concise replies")])

    # The active-entry threshold is not crossed by this one-entry update.
    assert len(calls) == 1
    assert chat.compactor is not None
    assert chat.compactor.metrics.attempted_calls == 0


def test_chat_records_token_usage_per_role():
    from memory_agent.application.chat import build_chat_memory

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

    from memory_agent.application.chat import build_chat_memory

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

    # The first call extracts the durable update and the second is the
    # bounded compactor review triggered by the active-entry threshold.
    assert len(calls) == 2
    assert chat.compactor is not None
    assert chat.compactor.metrics.attempted_calls == 1


def test_chat_render_is_bounded_by_answer_memory_budget():
    from memory_agent.application.chat import build_chat_memory

    chat = build_chat_memory(
        ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        config=ProductMemoryConfig(answer_memory_token_budget=20),
        compact=False,
    )
    chat.memory.apply_ops_atomically(
        [
            {"op": "ADD", "section": "facts", "text": "A" * 40, "provenance": [1]},
            {"op": "ADD", "section": "facts", "text": "B" * 40, "provenance": [2]},
        ]
    )

    rendered = chat.render()

    assert len(rendered) // 4 <= 20
    assert "B" * 40 in rendered
    assert "A" * 40 not in rendered


def test_chat_render_accepts_a_bounded_per_call_budget_override():
    from memory_agent.application.chat import build_chat_memory

    chat = build_chat_memory(
        ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        config=ProductMemoryConfig(answer_memory_token_budget=20),
        compact=False,
    )
    chat.memory.apply_ops_atomically(
        [
            {"op": "ADD", "section": "facts", "text": "A" * 40, "provenance": [1]},
            {"op": "ADD", "section": "facts", "text": "B" * 40, "provenance": [2]},
        ]
    )

    default_rendered = chat.render()
    widened_rendered = chat.render(max_tokens=40)

    assert len(default_rendered) // 4 <= 20
    assert len(widened_rendered) // 4 <= 40
    assert "B" * 40 in widened_rendered
    assert "A" * 40 in widened_rendered
    assert "A" * 40 not in default_rendered


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_chat_render_rejects_non_positive_or_non_integer_budget(budget):
    from memory_agent.application.chat import build_chat_memory

    chat = build_chat_memory(
        ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        compact=False,
    )

    with pytest.raises(ValueError, match="max_tokens"):
        chat.render(max_tokens=budget)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_product_config_rejects_invalid_answer_memory_budget(budget):
    with pytest.raises(ValueError, match="answer_memory_token_budget"):
        ProductMemoryConfig(answer_memory_token_budget=budget)


def test_chat_update_failure_is_reported_without_mutating_memory():
    from memory_agent.application.chat import build_chat_memory

    def fail(*_args):
        raise RuntimeError("transport down")

    chat = build_chat_memory(ScriptedLLM(fail), compact=False)
    before = chat.memory.to_state()

    applied, rejected = chat.update([Turn(1, "user", "Remember the launch is blocked.")])

    assert applied == []
    assert rejected and "updater_failed" in rejected[0]["reason"]
    assert chat.memory.to_state() == before


def test_chat_update_retries_only_deferred_suffix_and_is_idempotent():
    from memory_agent.application.chat import build_chat_memory

    responses = iter(
        [
            '[{"op":"ADD","section":"facts","text":"saved prefix",'
            '"provenance":[1]}]',
            '[{"op":"UPDATE","id":"F999","text":"invalid",'
            '"provenance":[3]}]',
            '[{"op":"ADD","section":"facts","text":"saved suffix",'
            '"provenance":[3]}]',
        ]
    )
    calls = []

    def responder(system, messages):
        calls.append((system, messages))
        return next(responses)

    chat = build_chat_memory(
        ScriptedLLM(responder),
        config=ProductMemoryConfig(evicted_turn_token_budget=2),
        compact=False,
    )
    chat.updater.max_retries = 0
    chat.updater.token_estimator = lambda _text: 1
    turns = [
        Turn(1, "user", "oldest"),
        Turn(2, "user", "middle"),
        Turn(3, "user", "newest"),
    ]

    applied, rejected = chat.update(turns)

    assert [op["text"] for op in applied] == ["saved prefix"]
    assert rejected and "F999" in rejected[0]["reason"]
    assert [entry.text for entry in chat.memory.entries.values()] == ["saved prefix"]
    assert len(calls) == 2
    diagnostics = chat.update_diagnostics()
    assert diagnostics["submitted_turn_ids"] == [1, 2, 3]
    assert diagnostics["committed_turn_ids"] == [1, 2]
    assert diagnostics["deferred_turn_ids"] == [3]
    assert diagnostics["retained_deferred_turn_ids"] == [3]
    assert diagnostics["dropped_turn_ids"] == []
    assert diagnostics["attempt_count"] == 1

    # The explicit retry receives only the deferred suffix. The public facade
    # must not conceal an exhausted suffix failure by retrying in this call.
    applied, rejected = chat.update(turns)

    assert rejected == []
    assert [op["text"] for op in applied] == ["saved suffix"]
    assert [entry.text for entry in chat.memory.entries.values()] == [
        "saved prefix",
        "saved suffix",
    ]
    assert len(calls) == 3
    retry_content = calls[-1][0] + "\n" + "\n".join(
        str(message.get("content", "")) for message in calls[-1][1]
    )
    assert "newest" in retry_content
    assert "oldest" not in retry_content
    assert "middle" not in retry_content
    diagnostics = chat.update_diagnostics()
    assert diagnostics["submitted_turn_ids"] == [3]
    assert diagnostics["committed_turn_ids"] == [3]
    assert diagnostics["deferred_turn_ids"] == []
    assert diagnostics["retained_deferred_turn_ids"] == []

    assert chat.update(turns) == ([], [])
    assert len(calls) == 3


def test_chat_update_retains_deferred_suffix_until_a_later_public_call():
    from memory_agent.application.chat import build_chat_memory

    responses = iter(
        [
            '[{"op":"ADD","section":"facts","text":"saved prefix",'
            '"provenance":[1]}]',
            '[{"op":"UPDATE","id":"F999","text":"invalid",'
            '"provenance":[3]}]',
            '[{"op":"ADD","section":"facts","text":"saved suffix",'
            '"provenance":[3]},'
            '{"op":"ADD","section":"facts","text":"saved next",'
            '"provenance":[4]}]',
        ]
    )
    calls = []

    def responder(system, messages):
        calls.append((system, messages))
        return next(responses)

    chat = build_chat_memory(
        ScriptedLLM(responder),
        config=ProductMemoryConfig(evicted_turn_token_budget=2),
        compact=False,
    )
    chat.updater.max_retries = 0
    chat.updater.token_estimator = lambda _text: 1

    first_batch = [
        Turn(1, "user", "oldest"),
        Turn(2, "user", "middle"),
        Turn(3, "user", "newest"),
    ]
    applied, rejected = chat.update(first_batch)
    assert [op["text"] for op in applied] == ["saved prefix"]
    assert rejected

    # The next caller need not resend the failed batch: the facade retains it
    # and retries it before processing newly supplied turns.
    applied, rejected = chat.update([Turn(4, "user", "next")])
    assert rejected == []
    assert [op["text"] for op in applied] == ["saved suffix", "saved next"]
    assert [entry.text for entry in chat.memory.entries.values()] == [
        "saved prefix",
        "saved suffix",
        "saved next",
    ]
    assert len(calls) == 3
    retry_content = calls[-1][0] + "\n" + "\n".join(
        str(message.get("content", "")) for message in calls[-1][1]
    )
    assert "newest" in retry_content
    assert "next" in retry_content
    assert retry_content.index("newest") < retry_content.index("next")

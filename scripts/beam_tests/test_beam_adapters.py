from pathlib import Path

import pytest

from evaluation.beam import BeamChatCaseAdapter, compare_fixed_budget_runs
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.retrieval.context import (
    build_answer_memory_context,
)
from memory_agent.retrieval.selector import MemorySelector
from scripts.run_beam_case import build_answer_context, build_answer_context_result
from scripts.run_beam_case import update_chat_memory
from langchain_core.messages import AIMessage, HumanMessage


def test_beam_chat_adapter_outputs_public_chat_turns():
    turns = BeamChatCaseAdapter().adapt_messages(
        [{"id": 7, "role": "user", "content": "hello"}], case_id="2"
    )
    assert turns[0].role == "user"
    assert turns[0].content == "hello"
    assert turns[0].id == 1


def test_fixed_budget_comparison_separates_production_and_judge_costs():
    rows = [
        {"variant": variant, "case_id": "1", "question_id": "q1", "context_budget_tokens": 256, "actual_context_tokens": 200, "quality": quality, "tokens": {"updater": 10, "compactor": 5, "agent": 20, "judge": 100}}
        for variant, quality in (("baseline", .5), ("current", .75))
    ]
    result = compare_fixed_budget_runs(rows)
    assert [item["variant"] for item in result] == ["baseline", "current"]
    assert result[0]["production_tokens"] == 35
    assert result[0]["judge_tokens"] == 100
    assert result[0]["validated_profile"] == "chat"
    assert result[0]["agent_memory_validated"] is False


def test_fixed_budget_comparison_rejects_mismatched_question_sets():
    with pytest.raises(ValueError, match="same cases"):
        compare_fixed_budget_runs([
            {"variant": "a", "case_id": "1", "question_id": "q1", "context_budget_tokens": 256},
            {"variant": "b", "case_id": "1", "question_id": "q2", "context_budget_tokens": 256},
        ])


def test_beam_production_and_middleware_service_are_byte_identical():
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "The project uses SQLite.", "provenance": [1]},
        {"op": "ADD", "section": "decisions", "text": "The team chose Postgres.", "provenance": [2]},
    ])
    selector = MemorySelector()
    entries = selector.select_for_answer(
        memory=memory,
        query="Which database does the project use?",
        budget=500,
    )
    result = build_answer_memory_context(
        memory=memory,
        entries=entries,
    )
    middleware = type("Middleware", (), {"memory": memory, "memory_selector": selector})()
    beam = build_answer_context(
        middleware, [], [], 100, 100, 500, query="Which database does the project use?",
        answer_memory_selection="selector",
    )
    typed_beam = build_answer_context_result(
        middleware, [], [], 100, 100, 500, query="Which database does the project use?",
        answer_memory_selection="selector",
    )
    rendered = beam.split("Structured memory summary.\n", 1)[1].split("\n\n# Chronological Order", 1)[0]
    assert rendered == result.rendered_context
    assert result.selected_ids == ("D1", "F1")
    assert typed_beam.selected_ids == result.selected_ids
    assert typed_beam.rendered_context == beam


def test_answer_memory_selection_all_bypasses_selector_budget():
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops([
        {
            "op": "ADD",
            "section": "facts",
            "text": "The project implemented a Flask homepage route.",
            "provenance": [1],
        },
        {
            "op": "ADD",
            "section": "facts",
            "text": "The unrelated deployment guide contains extensive operational notes.",
            "provenance": [2],
        },
    ])
    middleware = type(
        "Middleware",
        (),
        {
            "memory": memory,
            "memory_selector": MemorySelector(),
        },
    )()

    full = build_answer_context_result(
        middleware, [], [], 100, 100, 1,
        query="Have I implemented a Flask route?",
        answer_memory_selection="all",
    )
    selected = build_answer_context_result(
        middleware, [], [], 100, 100, 1,
        query="Have I implemented a Flask route?",
        answer_memory_selection="selector",
    )

    assert full.selected_ids == ("F1", "F2")
    assert selected.selected_ids == ()
    assert "Flask homepage route" in full.rendered_context


def test_answer_memory_selection_rejects_unknown_mode():
    memory = Memory(sections=CHAT_SECTIONS)
    middleware = type(
        "Middleware",
        (),
        {"memory": memory, "memory_selector": MemorySelector()},
    )()

    with pytest.raises(ValueError, match="answer_memory_selection"):
        build_answer_context_result(
            middleware, [], [], 100, 100, 100,
            answer_memory_selection="unknown",
        )


def test_root_unit_tests_do_not_import_beam_evaluation():
    offenders = []
    for path in Path("tests").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "evaluation.beam" in text or "scripts.run_beam" in text:
            offenders.append(path)
    assert offenders == []


def test_update_chat_memory_reports_only_committed_turns_and_preserves_ids():
    class FakeChatMemory:
        def __init__(self):
            self.calls = []
            self.diagnostics = {
                "submitted_turn_ids": [],
                "committed_turn_ids": [],
                "deferred_turn_ids": [],
                "dropped_turn_ids": [],
                "status": "idle",
            }

        def update(self, turns):
            self.calls.append([turn.id for turn in turns])
            self.diagnostics = {
                "submitted_turn_ids": [turn.id for turn in turns],
                "committed_turn_ids": [turns[0].id],
                "deferred_turn_ids": [turn.id for turn in turns[1:]],
                "dropped_turn_ids": [],
                "status": "partial",
            }
            return ([{"op": "ADD"}], [])

        def update_diagnostics(self):
            return self.diagnostics.copy()

    chat = FakeChatMemory()
    batch = [HumanMessage(content="old"), AIMessage(content="new")]

    report = update_chat_memory(chat, batch, batch_index=3, turn_ids=[301, 302])

    assert chat.calls == [[301, 302]]
    assert report["submitted_turn_ids"] == [301, 302]
    assert report["committed_turn_ids"] == [301]
    assert report["deferred_turn_ids"] == [302]
    assert report["dropped_turn_ids"] == []


def test_update_chat_memory_ignores_empty_and_unsupported_messages():
    class FakeChatMemory:
        def update(self, turns):
            raise AssertionError("empty batches must not invoke ChatMemory")

    report = update_chat_memory(
        FakeChatMemory(),
        [HumanMessage(content=" ")],
        batch_index=1,
    )

    assert report["status"] == "empty"
    assert report["submitted_turn_ids"] == []

"""Production-style invariants for the chat API, without BEAM metadata."""

from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.retrieval.context import build_answer_memory_context
from memory_agent.retrieval.selector import MemorySelector
from memory_agent.update.compactor import MemoryCompactor
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _memory():
    return Memory(CHAT_SECTIONS, policy=CHAT_POLICY)


def _updater(script):
    return MemoryUpdater(ScriptedLLM(script), CHAT_SECTIONS, policy=CHAT_POLICY)


def test_unaccepted_assistant_proposal_is_not_written():
    updater = MemoryUpdater(
        ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        CHAT_SECTIONS,
        policy=CHAT_POLICY,
    )

    applied, rejected = updater.update(
        _memory(),
        [
            Turn(1, "user", "Could we use Redis?"),
            Turn(2, "assistant", "I suggest Redis for caching."),
        ],
    )

    assert applied == [] and rejected == []


def test_substantive_exchange_can_be_saved_as_one_progress_rollup():
    memory = _memory()
    updater = _updater(
        lambda *_: (
            '[{"op":"ADD","section":"progress",'
            '"text":"Discussion covered base-height and Heron area methods.",'
            '"provenance":[1,2]}]'
        )
    )

    applied, rejected = updater.update(
        memory,
        [
            Turn(1, "user", "We decided to use base-height and Heron's formula."),
            Turn(2, "assistant", "Use base times height or Heron's formula."),
        ],
    )

    assert applied and rejected == []
    active = [entry for entry in memory.entries.values() if entry.status == "active"]
    assert len(active) == 1
    assert active[0].section == "progress"
    assert active[0].provenance == [1, 2]


def test_explicit_user_acceptance_can_be_saved_as_decision():
    updater = _updater(
        lambda *_: '[{"op":"ADD","section":"decisions",'
        '"text":"Use Redis for caching.","provenance":[3]}]'
    )
    memory = _memory()

    applied, rejected = updater.update(
        memory,
        [Turn(3, "user", "We decided to use Redis for caching.")],
    )

    assert applied and rejected == []
    assert any(entry.section == "decisions" for entry in memory.entries.values())


def test_production_selection_is_invariant_to_benchmark_metadata():
    memory = _memory()
    memory.apply_ops(
        [
            {"op": "ADD", "section": "facts", "text": "API latency is 250ms.", "provenance": [1]},
            {
                "op": "ADD",
                "section": "preferences",
                "text": "User prefers short answers.",
                "provenance": [2],
            },
        ]
    )
    selector = MemorySelector(policy=CHAT_POLICY)

    def production(metadata):
        del metadata
        entries = selector.select_for_answer(
            memory=memory,
            query="What is the API latency?",
            budget=100,
        )
        return build_answer_memory_context(memory=memory, entries=entries).selected_ids

    assert production({"question_type": "summarization", "case_id": "1"}) == production(
        {"question_type": "abstention", "case_id": "999", "rubric": "secret"}
    )


def test_irrelevant_query_does_not_force_unbounded_memory():
    memory = _memory()
    for index in range(20):
        memory.apply_ops(
            [
                {
                    "op": "ADD",
                    "section": "facts",
                    "text": f"Project {index} uses database {index}.",
                    "provenance": [index + 1],
                }
            ]
        )
    selector = MemorySelector(policy=CHAT_POLICY)
    entries = selector.select_for_answer(
        memory=memory,
        query="What music should I play?",
        budget=20,
    )
    context = build_answer_memory_context(memory=memory, entries=entries)

    assert selector.token_estimator(context.rendered_context) <= 20


def test_chat_compactor_leaves_model_noop_candidates_unchanged():
    memory = _memory()
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "Project alpha API latency is 250ms.",
                "provenance": [1],
            },
            {
                "op": "ADD",
                "section": "facts",
                "text": "Project beta API latency is 400ms.",
                "provenance": [2],
            },
        ]
    )
    compactor = MemoryCompactor(
        ScriptedLLM(lambda *_: "[]"),
        CHAT_SECTIONS,
        policy=CHAT_POLICY,
    )

    assert compactor.compact(memory) == ([], [])
    assert compactor.metrics.attempted_calls == 1
    assert sum(entry.status == "active" for entry in memory.entries.values()) == 2


def test_chat_memory_rejects_exact_value_inventory_ops():
    memory = _memory()
    applied, rejected = memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "exact_values",
                "text": "Release date is 2026-09-01.",
                "provenance": [1],
            }
        ]
    )

    assert applied == []
    assert rejected
    assert memory.entries == {}

"""Production-style memory invariants with no BEAM adapter or LLM judge."""

from memory_agent.models.policy import get_memory_policy
from memory_agent.models.sections import PRACTICAL_SECTIONS
from memory_agent.models.transcript import Turn
from memory_agent.structured.answer_context import (
    AnswerContextBudget,
    AnswerContextConfig,
    build_answer_memory_context,
)
from memory_agent.structured.compactor import MemoryCompactor
from memory_agent.structured.memory import Memory
from memory_agent.structured.selector import MemorySelector
from memory_agent.structured.updater import MemoryUpdater
from memory_agent.profiles.chat.subject_normalizer import ChatSubjectNormalizer
from tests.fakes import ScriptedLLM


def _memory():
    policy = get_memory_policy("practical")
    return policy, Memory(PRACTICAL_SECTIONS, policy=policy)


def test_unaccepted_assistant_proposal_is_not_written():
    policy, memory = _memory()
    updater = MemoryUpdater(
        ScriptedLLM(lambda *_: (_ for _ in ()).throw(AssertionError("ordinary batch called LLM"))),
        PRACTICAL_SECTIONS,
        policy=policy,
    )

    applied, rejected = updater.update(memory, [
        Turn(1, "user", "Could we use Redis?"),
        Turn(2, "assistant", "I suggest Redis for caching."),
    ])

    assert applied == [] and rejected == [] and memory.entries == {}


def test_explicit_user_acceptance_can_be_saved_as_decision():
    policy, memory = _memory()
    updater = MemoryUpdater(
        ScriptedLLM(lambda *_: '[{"op":"ADD","section":"decisions",'
                                  '"text":"Use Redis for caching.","provenance":[3]}]'),
        PRACTICAL_SECTIONS,
        policy=policy,
    )

    updater.update(memory, [Turn(3, "user", "We decided to use Redis for caching.")])

    assert any(entry.section == "decisions" for entry in memory.entries.values())


def test_production_selection_is_invariant_to_benchmark_metadata():
    policy, memory = _memory()
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "API latency is 250ms.", "provenance": [1]},
        {"op": "ADD", "section": "preferences", "text": "User prefers short answers.", "provenance": [2]},
    ])
    selector = MemorySelector(policy=policy, pinned_sections=frozenset())

    def production(metadata):
        del metadata
        return build_answer_memory_context(
            query="What is the API latency?",
            memory=memory,
            config=AnswerContextConfig(selector),
            budget=AnswerContextBudget(100),
        ).selected_ids

    assert production({"question_type": "summarization", "case_id": "1"}) == production(
        {"question_type": "abstention", "case_id": "999", "rubric": "secret"}
    )


def test_irrelevant_query_does_not_force_unbounded_memory():
    policy, memory = _memory()
    for index in range(20):
        memory.apply_ops([{
            "op": "ADD", "section": "facts", "text": f"Project {index} uses database {index}.",
            "provenance": [index + 1],
        }])
    selector = MemorySelector(policy=policy, pinned_sections=frozenset())
    context = build_answer_memory_context(
        query="What music should I play?", memory=memory,
        config=AnswerContextConfig(selector), budget=AnswerContextBudget(20),
    )

    assert selector.token_estimator(context.rendered_context) <= 20


def test_similar_entities_are_not_automatically_sent_to_semantic_compaction():
    policy, memory = _memory()
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "Project alpha API latency is 250ms.", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "Project beta API latency is 400ms.", "provenance": [2]},
    ])
    compactor = MemoryCompactor(
        ScriptedLLM(lambda *_: (_ for _ in ()).throw(AssertionError("transport called"))),
        PRACTICAL_SECTIONS,
        policy=policy,
        enable_semantic_candidates=False,
    )

    assert compactor.compact(memory) == ([], [])
    assert sum(entry.status == "active" for entry in memory.entries.values()) == 2


def test_named_container_count_has_stable_identity_but_unanchored_count_does_not():
    normalizer = ChatSubjectNormalizer()

    old = normalizer.normalize("User has ~45 sources in Zotero.")
    new = normalizer.normalize("I've added 52 sources to my Zotero library.")

    assert old is not None and new is not None
    assert old[0].entity == "zotero source count"
    assert new[0].entity == "zotero source count"
    assert normalizer.normalize("The example has 12 widgets.") is None


def test_named_container_count_update_supersedes_older_active_value():
    policy, memory = _memory()
    updater = MemoryUpdater(
        ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        PRACTICAL_SECTIONS,
        policy=policy,
    )

    updater.update(memory, [Turn(1, "user", "I've added 45 sources to my Zotero library.")])
    updater.update(memory, [Turn(2, "user", "I now have 52 sources in Zotero.")])

    active = [entry for entry in memory.entries.values() if entry.status == "active"]
    superseded = [entry for entry in memory.entries.values() if entry.status == "superseded"]
    assert len(active) == 1 and active[0].value.value == "52"
    assert len(superseded) == 1
    assert active[0].provenance == [1, 2]

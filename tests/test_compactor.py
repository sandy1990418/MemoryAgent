import json

from memory_agent.models.policy import get_memory_policy
from memory_agent.models.sections import PRACTICAL_SECTIONS
from memory_agent.models.memory import MemoryValue, SubjectIdentity
from memory_agent.structured.compactor import CompactionCandidate, MemoryCompactor
from memory_agent.structured.memory import Memory
from tests.fakes import ScriptedLLM
from tests.practical_cases import SUBJECT_COMPACTION_CASE


def test_compaction_reduces_active_entries_and_preserves_latest_truth():
    case = SUBJECT_COMPACTION_CASE
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    applied, rejected = memory.apply_ops(case["preload"])
    assert rejected == []
    assert applied
    active_before = sum(entry.status == "active" for entry in memory.entries.values())

    compactor = MemoryCompactor(
        llm=ScriptedLLM(
            lambda system, messages: json.dumps(case["compactor_ops"])
        ),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    applied, rejected = compactor.compact(memory)

    assert rejected == []
    assert applied
    active_after = sum(entry.status == "active" for entry in memory.entries.values())
    assert active_after < active_before
    active_texts = {
        entry.text for entry in memory.entries.values() if entry.status == "active"
    }
    assert active_texts == set(case["expected_active"])
    assert "summary-based" in next(iter(active_texts))
    assert "rather than mem0" in next(iter(active_texts))
    assert memory.entries["D1"].status == "superseded"
    assert memory.entries["D2"].status == "superseded"
    assert memory.entries["D3"].status == "superseded"


def test_compactor_rejects_reactivating_superseded_entry():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "decisions",
                "text": "Use mem0 for product memory.",
                "provenance": [1],
            },
            {"op": "SUPERSEDE", "id": "D1", "reason": "Decision changed."},
            {
                "op": "ADD",
                "section": "decisions",
                "text": "Use summary memory.",
                "provenance": [2],
            },
            {
                "op": "ADD",
                "section": "decisions",
                "text": "Do not use mem0.",
                "provenance": [3],
            },
        ]
    )
    ops = [
        {"op": "SUPERSEDE", "id": "D2", "reason": "Bad merge."},
        {"op": "SUPERSEDE", "id": "D3", "reason": "Bad merge."},
        {
            "op": "ADD",
            "section": "decisions",
            "text": "Use mem0 for product memory.",
            "provenance": [2, 3],
        },
    ]
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda system, messages: json.dumps(ops)),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )

    applied, rejected = compactor.compact(memory)

    assert applied == []
    assert rejected
    assert memory.entries["D2"].status == "active"
    assert memory.entries["D3"].status == "active"


def test_compactor_accepts_string_noop_from_small_models():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "Project uses Flask.",
                "provenance": [1],
            }
        ]
    )
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda system, messages: '["NOOP"]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )

    applied, rejected = compactor.compact(memory)

    assert applied == []
    assert rejected == []


def test_compactor_normalizes_small_model_key_value_source_id_schema():
    policy = get_memory_policy("chat")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "API latency was 300ms.", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "API latency is now 250ms.", "provenance": [2]},
    ])
    response = json.dumps([
        {"op": "SUPERSEDE", "key": "facts", "source_provenance_ids": ["F1", "F2"], "reason": "Merged."},
        {"op": "ADD", "key": "facts", "value": "API latency is now 250ms.", "source_provenance_ids": ["F1", "F2"]},
    ])
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda system, messages: response),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    applied, rejected = compactor.compact(memory)
    assert rejected == []
    assert len(applied) == 3
    assert memory.entries["F1"].status == "superseded"
    assert memory.entries["F2"].status == "superseded"
    assert memory.entries["F3"].provenance == [1, 2]


def test_compactor_normalizes_camel_case_and_nested_value_schema():
    policy = get_memory_policy("chat")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "goal", "text": "Ship MVP by April 15.", "provenance": [1]},
        {"op": "ADD", "section": "goal", "text": "MVP includes login and analytics.", "provenance": [2]},
    ])
    response = json.dumps([
        {"op": "SUPERSEDE", "key": "goal", "sourceProvenanceIds": ["G1", "G2"]},
        {"op": "ADD", "section": "goal", "value": {"description": "Ship login and analytics MVP by April 15."}, "provenance_ids": ["G1", "G2"]},
    ])
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda system, messages: response),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    applied, rejected = compactor.compact(memory)
    assert rejected == []
    assert len(applied) == 3
    assert memory.entries["G3"].text == "Ship login and analytics MVP by April 15."
    assert memory.entries["G3"].provenance == [1, 2]


def test_candidate_prompt_never_contains_superseded_or_unrelated_entries():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "API latency is 300ms.", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "API latency is 250ms.", "provenance": [2]},
        {"op": "ADD", "section": "goal", "text": "SECRET unrelated goal", "provenance": [3]},
        {"op": "ADD", "section": "facts", "text": "SECRET historical value", "provenance": [4]},
        {"op": "SUPERSEDE", "id": "F3", "reason": "old"},
    ])
    seen = {}
    llm = ScriptedLLM(lambda system, messages: seen.setdefault("system", system) or "[]")
    compactor = MemoryCompactor(llm=llm, sections=PRACTICAL_SECTIONS, policy=policy)
    candidate = CompactionCandidate("api-latency", (memory.entries["F1"], memory.entries["F2"]), "semantic")
    compactor.compact_candidates(memory, [candidate])
    assert "API latency" in seen["system"]
    assert "SECRET" not in seen["system"]


def test_hidden_id_rejection_is_atomic_and_categorized():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "API latency is 300ms.", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "API latency is 250ms.", "provenance": [2]},
        {"op": "ADD", "section": "facts", "text": "Hidden fact.", "provenance": [3]},
    ])
    before = repr(memory.entries)
    response = json.dumps([
        {"op": "SUPERSEDE", "id": "F1"}, {"op": "SUPERSEDE", "id": "F3"},
        {"op": "ADD", "section": "facts", "text": "API latency is 250ms.", "provenance": [1, 3]},
    ])
    compactor = MemoryCompactor(llm=ScriptedLLM(lambda *_: response), sections=PRACTICAL_SECTIONS, policy=policy)
    candidate = CompactionCandidate("api", (memory.entries["F1"], memory.entries["F2"]), "semantic")
    applied, rejected = compactor.compact_candidates(memory, [candidate])
    assert applied == [] and rejected[0]["reason"] == "hidden_id"
    assert repr(memory.entries) == before
    assert compactor.metrics.attempted_calls == 1
    assert compactor.metrics.rejected_compactions == 1


def test_typed_subject_compacts_deterministically_without_transport():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    identity = SubjectIdentity("project", "api", "latency")
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "API latency is 300ms.", "provenance": [1], "subject_identity": identity, "value": MemoryValue("300", "ms")},
        {"op": "ADD", "section": "facts", "text": "API latency is 250ms.", "provenance": [2], "subject_identity": identity, "value": MemoryValue("250", "ms")},
    ])
    compactor = MemoryCompactor(llm=ScriptedLLM(lambda *_: (_ for _ in ()).throw(AssertionError("transport called"))), sections=PRACTICAL_SECTIONS, policy=policy)
    applied, rejected = compactor.compact(memory)
    assert len(applied) == 3 and rejected == []
    assert compactor.metrics.attempted_calls == 0
    assert compactor.metrics.deterministic_compactions == 1
    assert compactor.metrics.before_active == 2 and compactor.metrics.after_active == 1
    assert memory.entries["F3"].provenance == [1, 2]


def test_typed_subject_with_incompatible_units_or_qualifiers_is_not_a_candidate():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    base = dict(namespace="project", entity="api", attribute="latency")
    memory.apply_ops([
        {"op":"ADD", "section":"facts", "text":"Online API latency is 250ms.", "provenance":[1], "subject_identity":SubjectIdentity(**base, qualifier="online"), "value":MemoryValue("250", "ms")},
        {"op":"ADD", "section":"facts", "text":"Offline API latency is 2s.", "provenance":[2], "subject_identity":SubjectIdentity(**base, qualifier="offline"), "value":MemoryValue("2", "s")},
    ])
    compactor = MemoryCompactor(llm=ScriptedLLM(lambda *_: (_ for _ in ()).throw(AssertionError("transport called"))), sections=PRACTICAL_SECTIONS, policy=policy)
    assert compactor.detect_candidates(memory) == []
    assert compactor.compact(memory) == ([], [])
    assert sum(entry.status == "active" for entry in memory.entries.values()) == 2


def test_production_mode_disables_ambiguous_semantic_candidates():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "Flask app uses SQLite.", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "Flask app uses Bootstrap.", "provenance": [2]},
    ])
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda *_: (_ for _ in ()).throw(AssertionError("transport called"))),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
        enable_semantic_candidates=False,
    )

    assert compactor.detect_candidates(memory) == []


def test_candidate_budget_is_enforced_before_transport():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "A " * 50, "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "B " * 50, "provenance": [2]},
    ])
    compactor = MemoryCompactor(llm=ScriptedLLM(lambda *_: "[]"), sections=PRACTICAL_SECTIONS, policy=policy, max_candidate_tokens=1)
    candidate = CompactionCandidate("large", (memory.entries["F1"], memory.entries["F2"]), "semantic")
    _, rejected = compactor.compact_candidates(memory, [candidate])
    assert rejected[0]["reason"] == "budget"
    assert compactor.metrics.attempted_calls == 0


def test_record_skip_is_observable_without_attempting_transport():
    policy = get_memory_policy("practical")
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda *_: "[]"),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )

    compactor.record_skip("circuit_breaker")

    assert compactor.metrics.skipped_compactions == 1
    assert compactor.metrics.attempted_calls == 0
    assert compactor.metrics.failure_reasons == {"circuit_breaker": 1}

import json

from memory_agent.core.models import MemoryValue, SubjectIdentity
from memory_agent.core.sections import PRACTICAL_SECTIONS, sections_for_preset
from memory_agent.core.store import Memory
from memory_agent.policies.structured import get_memory_policy
from memory_agent.update.compactor import CompactionCandidate, MemoryCompactor
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
                "text": "Use summary-based product memory.",
                "provenance": [2],
            },
            {
                "op": "ADD",
                "section": "decisions",
                "text": "Product memory must not use mem0.",
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


def test_exactly_two_unrelated_entries_do_not_create_candidate():
    policy = get_memory_policy("chat")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "The sky is blue.", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "Redis runs locally.", "provenance": [2]},
    ])
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda *_: "[]"), sections=PRACTICAL_SECTIONS,
        policy=policy, enable_semantic_candidates=True,
    )
    assert compactor.detect_candidates(memory) == []


def test_related_progress_is_rolled_up_by_topic_and_preserves_provenance():
    policy = get_memory_policy("chat")
    sections = sections_for_preset("chat")
    memory = Memory(sections=sections, policy=policy)
    memory.apply_ops([
        {
            "op": "ADD",
            "section": "progress",
            "text": "Triangle work compared base-height and Heron area methods.",
            "provenance": [1, 2],
        },
        {
            "op": "ADD",
            "section": "progress",
            "text": "Triangle work derived the median formula and equal-area property.",
            "provenance": [3, 4],
        },
        {
            "op": "ADD",
            "section": "progress",
            "text": "Triangle work progressed through SSS, SAS, and ASA comparisons.",
            "provenance": [5, 6],
        },
    ])
    response = json.dumps([
        {"op": "SUPERSEDE", "id": "P1", "reason": "Topic rollup."},
        {"op": "SUPERSEDE", "id": "P2", "reason": "Topic rollup."},
        {"op": "SUPERSEDE", "id": "P3", "reason": "Topic rollup."},
        {
            "op": "ADD",
            "section": "progress",
            "text": (
                "Triangle study progressed from area methods to median properties, "
                "then compared SSS, SAS, and ASA congruence criteria."
            ),
            "provenance": [1, 2, 3, 4, 5, 6],
        },
    ])
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda *_: response),
        sections=sections,
        policy=policy,
        enable_semantic_candidates=False,
    )

    candidates = compactor.detect_candidates(memory)
    assert len(candidates) == 1
    assert candidates[0].reason == "progress-rollup"

    applied, rejected = compactor.compact_candidates(memory, candidates)

    assert len(applied) == 4 and rejected == []
    active = [entry for entry in memory.entries.values() if entry.status == "active"]
    assert len(active) == 1
    assert active[0].id == "P4"
    assert active[0].provenance == [1, 2, 3, 4, 5, 6]
    assert all(entry_id not in memory.entries for entry_id in ("P1", "P2", "P3"))


def test_progress_rollup_ignores_model_managed_source_ids():
    policy = get_memory_policy("chat")
    sections = sections_for_preset("chat")
    memory = Memory(sections=sections, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "progress", "text": "Flask auth covered login routes.", "provenance": [1, 2]},
        {"op": "ADD", "section": "progress", "text": "Flask auth added CSRF handling.", "provenance": [3, 4]},
        {"op": "ADD", "section": "progress", "text": "Flask auth compared password hashing.", "provenance": [5, 6]},
        {"op": "ADD", "section": "progress", "text": "Flask auth documented session security.", "provenance": [7, 8]},
    ])
    # Small models sometimes hallucinate an id outside the visible candidate.
    # The summary is useful, but source lifecycle must remain deterministic.
    response = json.dumps([
        {"op": "SUPERSEDE", "id": "P99", "reason": "Merged."},
        {
            "op": "ADD",
            "section": "progress",
            "text": "Flask authentication progressed through routes, CSRF, hashing, and session security.",
            "provenance": [999],
        },
    ])
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda *_: response),
        sections=sections,
        policy=policy,
        enable_semantic_candidates=False,
    )

    applied, rejected = compactor.compact(memory)

    assert rejected == []
    assert len(applied) == 5
    assert memory.entries["P5"].provenance == list(range(1, 9))
    assert all(f"P{index}" not in memory.entries for index in range(1, 5))


def test_large_progress_backlog_is_split_into_bounded_rollup_candidates():
    policy = get_memory_policy("chat")
    sections = sections_for_preset("chat")
    memory = Memory(sections=sections, policy=policy)
    for index in range(12):
        memory.apply_ops([{
            "op": "ADD",
            "section": "progress",
            "text": f"Flask project authentication topic step {index + 1}.",
            "provenance": [index * 2 + 1, index * 2 + 2],
        }])
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda *_: "[]"),
        sections=sections,
        policy=policy,
        enable_semantic_candidates=False,
        max_candidate_entries=8,
    )

    candidates = compactor.detect_candidates(memory)

    assert [len(candidate.entries) for candidate in candidates] == [8, 4]
    assert all(candidate.reason == "progress-rollup" for candidate in candidates)


def test_progress_rollup_accepts_plain_text_and_replaces_only_candidate_entries():
    policy = get_memory_policy("chat")
    sections = sections_for_preset("chat")
    memory = Memory(sections=sections, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "progress", "text": "Flask authentication discussion covered route protection, login redirects, and session loading behavior in the existing application structure.", "provenance": [1, 2]},
        {"op": "ADD", "section": "progress", "text": "Flask authentication discussion covered CSRF validation, secure cookie settings, and password-hash verification during login.", "provenance": [3, 4]},
        {"op": "ADD", "section": "progress", "text": "Flask authentication discussion covered RBAC decorators, unauthorized responses, and tests for admin and user endpoints.", "provenance": [5, 6]},
        {"op": "ADD", "section": "progress", "text": "Triangle geometry covered Heron's formula and medians.", "provenance": [7, 8]},
    ])
    plain_summary = (
        "Flask authentication progressed through route and session integration, "
        "CSRF and secure-cookie handling, password verification, RBAC decorators, "
        "unauthorized behavior, and endpoint tests."
    )
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda *_: plain_summary),
        sections=sections,
        policy=policy,
        enable_semantic_candidates=False,
    )
    candidate = CompactionCandidate(
        "progress-rollup:P1,P2,P3",
        (memory.entries["P1"], memory.entries["P2"], memory.entries["P3"]),
        "progress-rollup",
    )

    applied, rejected = compactor.compact_candidates(memory, [candidate])

    assert rejected == [] and len(applied) == 4
    assert memory.entries["P4"].status == "active"
    assert memory.entries["P5"].text == plain_summary
    assert memory.entries["P5"].provenance == [1, 2, 3, 4, 5, 6]
    assert set(memory.entries) == {"P4", "P5"}


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


def test_generic_typed_identity_is_not_a_safe_compaction_candidate():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    generic_goal = SubjectIdentity("chat", "goal", "goal", confidence=0.9)
    memory.apply_ops([
        {"op":"ADD", "section":"facts", "text":"Emergency fund goal is $2,000.", "provenance":[1], "subject_identity":generic_goal, "value":MemoryValue("2000", "$")},
        {"op":"ADD", "section":"facts", "text":"Family car goal is $5,000.", "provenance":[2], "subject_identity":generic_goal, "value":MemoryValue("5000", "$")},
    ])
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda *_: (_ for _ in ()).throw(AssertionError("transport called"))),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )

    assert compactor.detect_candidates(memory) == []
    assert compactor.compact(memory) == ([], [])


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

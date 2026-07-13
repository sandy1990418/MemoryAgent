import json

from memory_agent.adapters.langchain.structured_memory import StructuredMemoryMiddleware
from memory_agent.core.sections import CHAT_SECTIONS, sections_for_preset
from memory_agent.core.store import Memory
from memory_agent.policies.structured import get_memory_policy
from memory_agent.update.compactor import MemoryCompactor
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _middleware(*, threshold=1, enabled=True):
    policy = get_memory_policy("chat")
    llm = ScriptedLLM(lambda *_: "[]")
    memory = Memory(CHAT_SECTIONS, policy=policy)
    updater = MemoryUpdater(llm, CHAT_SECTIONS, policy=policy)
    compactor = MemoryCompactor(
        llm, CHAT_SECTIONS, policy=policy, enable_semantic_candidates=False
    ) if enabled else None
    return StructuredMemoryMiddleware(
        memory, updater, max_tokens=100, policy=policy, compactor=compactor,
        compact_min_active_entries=threshold,
    )


def test_compactor_diagnostics_distinguish_disabled_below_and_no_candidates():
    disabled = _middleware(enabled=False)
    disabled._maybe_compact()
    assert disabled.compaction_diagnostics()["skip_reasons"] == {"disabled_profile": 1}

    below = _middleware(threshold=2)
    below.memory.apply_ops([{"op":"ADD", "section":"facts", "text":"One fact.", "provenance":[1]}])
    below._maybe_compact()
    assert below.compaction_diagnostics()["skip_reasons"] == {"below_scan_threshold": 1}

    none = _middleware(threshold=1)
    none.memory.apply_ops([
        {"op":"ADD", "section":"facts", "text":"One fact.", "provenance":[1]},
        {"op":"ADD", "section":"facts", "text":"Redis runs locally.", "provenance":[2]},
    ])
    none._maybe_compact()
    report = none.compaction_diagnostics()
    assert report["skip_reasons"] == {"no_candidates": 1}
    assert report["checks"][0]["active_entries_at_check"] == 2
    assert report["checks"][0]["total_entries_at_check"] == 2
    assert report["attempted_calls"] == 0


def test_candidate_presence_controls_deterministic_compaction():
    middleware = _middleware(threshold=1)
    middleware.memory.apply_ops([
        {"op":"ADD", "section":"facts", "text":"Same fact.", "provenance":[1]},
        {"op":"ADD", "section":"facts", "text":"Same fact.", "provenance":[2]},
    ])
    middleware._maybe_compact()
    report = middleware.compaction_diagnostics()
    assert report["checks"][0]["skip_reason"] == "deterministic_candidate"
    assert middleware.compactor.metrics.deterministic_compactions == 1
    assert report["attempted_calls"] == 0


def test_progress_rollup_bypasses_global_scan_threshold():
    policy = get_memory_policy("chat")
    sections = sections_for_preset("chat")
    memory = Memory(sections, policy=policy)
    updater = MemoryUpdater(ScriptedLLM(lambda *_: "[]"), sections, policy=policy)
    response = json.dumps([
        {"op": "SUPERSEDE", "id": "P1", "reason": "Topic rollup."},
        {"op": "SUPERSEDE", "id": "P2", "reason": "Topic rollup."},
        {"op": "SUPERSEDE", "id": "P3", "reason": "Topic rollup."},
        {
            "op": "ADD",
            "section": "progress",
            "text": "Triangle work progressed from area methods to medians and congruence.",
            "provenance": [1, 2, 3, 4, 5, 6],
        },
    ])
    compactor = MemoryCompactor(
        ScriptedLLM(lambda *_: response),
        sections,
        policy=policy,
        enable_semantic_candidates=False,
    )
    middleware = StructuredMemoryMiddleware(
        memory,
        updater,
        max_tokens=100,
        policy=policy,
        compactor=compactor,
        compact_min_active_entries=30,
    )
    memory.apply_ops([
        {"op": "ADD", "section": "progress", "text": "Triangle area used Heron's formula.", "provenance": [1, 2]},
        {"op": "ADD", "section": "progress", "text": "Triangle medians split equal areas.", "provenance": [3, 4]},
        {"op": "ADD", "section": "progress", "text": "Triangle congruence compared SAS and ASA.", "provenance": [5, 6]},
    ])

    middleware._maybe_compact()

    report = middleware.compaction_diagnostics()
    assert report["checks"][0]["active_entries_at_check"] == 3
    assert report["checks"][0]["skip_reason"] == "llm_candidate"
    assert report["attempted_calls"] == 1
    assert sum(entry.status == "active" for entry in memory.entries.values()) == 1

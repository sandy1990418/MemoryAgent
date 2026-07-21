from memory_agent.adapters.langchain.chat_memory import LangChainChatAdapter
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.update.compactor import MemoryCompactor
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _middleware(*, threshold=1, enabled=True):
    llm = ScriptedLLM(lambda *_: "[]")
    memory = Memory(CHAT_SECTIONS, policy=CHAT_POLICY)
    updater = MemoryUpdater(llm, CHAT_SECTIONS, policy=CHAT_POLICY)
    compactor = (
        MemoryCompactor(
            llm,
            CHAT_SECTIONS,
            policy=CHAT_POLICY,
        )
        if enabled
        else None
    )
    return LangChainChatAdapter(
        memory,
        updater,
        max_tokens=100,
        policy=CHAT_POLICY,
        compactor=compactor,
        compact_min_active_entries=threshold,
    )


def test_compactor_diagnostics_distinguish_disabled_below_and_no_candidates():
    disabled = _middleware(enabled=False)
    disabled._maybe_compact()
    assert disabled.compaction_diagnostics()["skip_reasons"] == {"disabled": 1}

    below = _middleware(threshold=2)
    below.memory.apply_ops(
        [{"op": "ADD", "section": "facts", "text": "One fact.", "provenance": [1]}]
    )
    below._maybe_compact()
    assert below.compaction_diagnostics()["skip_reasons"] == {"below_scan_threshold": 1}

    none = _middleware(threshold=0)
    none.memory.apply_ops(
        [{"op": "ADD", "section": "facts", "text": "One fact.", "provenance": [1]}]
    )
    none._maybe_compact()
    report = none.compaction_diagnostics()
    assert report["skip_reasons"] == {"no_candidates": 1}
    assert report["checks"][0]["active_entries_at_check"] == 1
    assert report["checks"][0]["total_entries_at_check"] == 1
    assert report["attempted_calls"] == 0

"""Prompt safety contracts for the single chat updater."""

from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.update.compactor import MemoryCompactor
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _llm():
    return ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]')


def _updater():
    return MemoryUpdater(llm=_llm(), sections=CHAT_SECTIONS, policy=CHAT_POLICY)


def _memory():
    return Memory(sections=CHAT_SECTIONS, policy=CHAT_POLICY)


def test_chat_prompt_has_one_consistent_batch_limit():
    system, messages = _updater()._build_prompt(
        _memory(),
        [Turn(id=7, role="user", content="I changed my mind: use SQLite 3.39.")],
    )

    assert "at most three concise ADD or UPDATE operations" in system
    assert "at most one durable ADD or UPDATE" not in system
    assert "EVAL PROFILE" not in system
    assert messages == [
        {
            "role": "user",
            "content": "Apply the rules above and return the ops JSON array for these turns.",
        }
    ]


def test_updater_prompt_omits_code_payload_and_bounds_pathological_turns():
    content = "User completed login.\n```python\n" + ("print('noise')\n" * 3000) + "```\nFinal constraint."
    system, _messages = _updater()._build_prompt(
        _memory(),
        [Turn(id=1, role="user", content=content)],
    )

    assert "[code block omitted from memory extraction]" in system
    assert "print('noise')" not in system
    assert "User completed login" in system
    assert "Final constraint" in system


def test_compactor_prompt_preserves_history_and_canonical_entry_rules():
    memory = _memory()
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
        llm=_llm(),
        sections=CHAT_SECTIONS,
        policy=CHAT_POLICY,
    )

    system, messages = compactor._build_prompt(memory)

    assert "SUPERSEDE every replaced active entry" in system
    assert "Canonical ADD provenance must be the union" in system
    assert "Never operate on or re-activate a superseded entry" in system
    assert messages[0]["content"].startswith("Return subject-based compaction")

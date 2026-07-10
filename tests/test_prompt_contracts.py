from memory_agent.models.policy import get_memory_policy
from memory_agent.models.sections import EVAL_SECTIONS, PRACTICAL_SECTIONS
from memory_agent.models.transcript import Turn
from memory_agent.structured.compactor import MemoryCompactor
from memory_agent.structured.memory import Memory
from memory_agent.structured.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _llm():
    return ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]')


def _updater_prompt(profile, sections):
    policy = get_memory_policy(profile)
    updater = MemoryUpdater(llm=_llm(), sections=sections, policy=policy)
    memory = Memory(sections=sections, policy=policy)
    return updater._build_prompt(
        memory,
        [Turn(id=7, role="user", content="I changed my mind: use SQLite 3.39.")],
    )


def test_practical_prompt_has_one_consistent_batch_limit():
    system, messages = _updater_prompt("practical", PRACTICAL_SECTIONS)

    assert "at most three concise ADD or UPDATE operations" in system
    assert "at most one durable ADD or UPDATE" not in system
    assert messages == [
        {
            "role": "user",
            "content": "Apply the rules above and return the ops JSON array for these turns.",
        }
    ]


def test_eval_prompt_preserves_detailed_evaluation_rules():
    system, _messages = _updater_prompt("eval", EVAL_SECTIONS)

    assert "EVAL PROFILE" in system
    assert "For knowledge updates, keep the latest value active" in system
    assert "For information extraction, keep granular subject-bound facts" in system


def test_compactor_prompt_preserves_history_and_canonical_entry_rules():
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
        llm=_llm(),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )

    system, messages = compactor._build_prompt(memory)

    assert "SUPERSEDE every replaced active entry" in system
    assert "Canonical ADD provenance must be the union" in system
    assert "Never operate on or re-activate a superseded entry" in system
    assert messages[0]["content"].startswith("Return subject-based compaction")

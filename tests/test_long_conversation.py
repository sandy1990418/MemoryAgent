import json
import re

from memory_agent.models.sections import CHAT_SECTIONS
from memory_agent.structured.session import MemorySession
from memory_agent.structured.updater import MemoryUpdater
from tests.fakes import ScriptedLLM

DECISION_MARKER = "DECISION: use in-memory storage"
CHANGE_MARKER = "CHANGE: we now use file storage instead"

ADD_TEXT_INITIAL = "we decided to use in-memory storage for the cache layer"
ADD_TEXT_UPDATED = "we now use file storage for the cache layer"


def evicted_turn_id_containing(system: str, marker: str) -> int | None:
    prompt_marker = "Turns JSON to process:\n"
    turns_json = system.split(prompt_marker, 1)[1]
    turns = json.loads(turns_json)
    for turn in turns:
        if marker in turn["content"]:
            return turn["turn_id"]
    return None


def updater_script(system: str, messages: list[dict]) -> str:
    """Deterministic fake updater: scans the evicted-turn text embedded in
    the system prompt for known markers and returns the corresponding ops.
    """
    ops = []

    decision_turn_id = evicted_turn_id_containing(system, DECISION_MARKER)
    if decision_turn_id is not None:
        ops.append(
            {
                "op": "ADD",
                "section": "decisions",
                "text": ADD_TEXT_INITIAL,
                "provenance": [decision_turn_id],
            }
        )

    change_turn_id = evicted_turn_id_containing(system, CHANGE_MARKER)
    if change_turn_id is not None:
        old_id_match = re.search(r"\[(D\d+)\] " + re.escape(ADD_TEXT_INITIAL), system)
        if old_id_match:
            ops.append(
                {
                    "op": "SUPERSEDE",
                    "id": old_id_match.group(1),
                    "reason": "switched to file storage",
                }
            )
        ops.append(
            {
                "op": "ADD",
                "section": "decisions",
                "text": ADD_TEXT_UPDATED,
                "provenance": [change_turn_id],
            }
        )

    if not ops:
        ops = [{"op": "NOOP"}]

    return json.dumps(ops)


def build_prompts(num_sends: int, decision_index: int, change_index: int) -> list[str]:
    prompts = []
    padding = "padding words to inflate the token estimate for eviction purposes. "
    for i in range(num_sends):
        if i == decision_index:
            prompts.append(f"{DECISION_MARKER} for our cache layer. {padding}")
        elif i == change_index:
            prompts.append(f"{CHANGE_MARKER} of in-memory storage, for durability. {padding}")
        elif i == num_sends - 1:
            prompts.append(
                "Can you remind me what we decided about storage early in this "
                "conversation, and is it still true?"
            )
        else:
            prompts.append(f"filler message number {i}. {padding}{padding}")
    return prompts


def test_long_conversation_survives_eviction_and_handles_conflict():
    num_sends = 200
    decision_index = 1
    change_index = 60
    max_window_tokens = 300

    chat_llm = ScriptedLLM(lambda system, messages: "ok")
    updater_llm = ScriptedLLM(updater_script)
    updater = MemoryUpdater(llm=updater_llm, sections=CHAT_SECTIONS)

    session = MemorySession(
        chat_llm=chat_llm,
        updater=updater,
        sections=CHAT_SECTIONS,
        max_window_tokens=max_window_tokens,
    )

    prompts = build_prompts(num_sends, decision_index, change_index)

    max_observed_tokens_after_warmup = 0
    for i, prompt in enumerate(prompts):
        session.send(prompt)
        if i > 5:
            max_observed_tokens_after_warmup = max(
                max_observed_tokens_after_warmup, session.window.total_tokens()
            )

    # (1) final system prompt reflects the file-storage decision.
    assert ADD_TEXT_UPDATED in session.last_system_prompt

    # (2) the original in-memory decision entry still exists, superseded.
    superseded_entries = [
        e
        for e in session.memory.entries.values()
        if e.status == "superseded" and e.text == ADD_TEXT_INITIAL
    ]
    assert len(superseded_entries) == 1

    active_entries = [
        e
        for e in session.memory.entries.values()
        if e.status == "active" and e.text == ADD_TEXT_UPDATED
    ]
    assert len(active_entries) == 1

    # (3) transcript has every turn (user + assistant per send).
    assert len(session.transcript) == num_sends * 2

    # (4) window token total stays roughly within budget throughout
    #     (small overshoot from the trailing assistant reply is expected).
    assert max_observed_tokens_after_warmup <= max_window_tokens + 10

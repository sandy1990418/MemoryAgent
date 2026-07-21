"""End-to-end token-window behavior for the framework-free chat session."""

import json

from memory_agent.application.session import MemorySession
from memory_agent.core.sections import CHAT_SECTIONS
from tests.fakes import ScriptedLLM
from memory_agent.update.updater import MemoryUpdater


def updater_script(system: str, messages: list[dict]) -> str:
    turns_json = system.split("Turns JSON:\n", 1)[1]
    turns = json.loads(turns_json)
    user_turns = [turn for turn in turns if turn["role"] == "user"]
    if not user_turns:
        return "[{\"op\":\"NOOP\"}]"
    turn = user_turns[-1]
    return json.dumps(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": f"Conversation retained: {turn['content'][:80]}",
                "provenance": [turn["turn_id"]],
            }
        ]
    )


def test_long_conversation_survives_eviction_and_keeps_window_bounded():
    num_sends = 40
    max_window_tokens = 80
    session = MemorySession(
        chat_llm=ScriptedLLM(lambda *_: "ok"),
        updater=MemoryUpdater(
            llm=ScriptedLLM(updater_script),
            sections=CHAT_SECTIONS,
        ),
        sections=CHAT_SECTIONS,
        max_window_tokens=max_window_tokens,
    )

    for index in range(num_sends):
        session.send(f"Durable project update number {index}: the build is healthy.")

    active_entries = [
        entry for entry in session.memory.entries.values() if entry.status == "active"
    ]

    assert active_entries
    assert len(session.transcript) == num_sends * 2
    assert session.window.total_tokens() <= max_window_tokens + 10
    assert session.last_system_prompt.startswith("You are a helpful assistant.")

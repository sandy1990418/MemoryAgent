"""Frozen memory snapshot write/replay used for paired BEAM comparisons."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from evaluation.beam.memory_snapshot import (
    load_memory_snapshot,
    restore_from_snapshot,
    write_memory_snapshot,
)
from memory_agent.core.sections import PRACTICAL_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.policies.structured import get_memory_policy


def _memory() -> Memory:
    memory = Memory(PRACTICAL_SECTIONS, policy=get_memory_policy("practical"))
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "API latency is 250ms.", "provenance": [1]},
        {"op": "ADD", "section": "preferences", "text": "User prefers short answers.", "provenance": [2]},
    ])
    return memory


def _messages() -> list:
    return [
        HumanMessage(content="What is the latency?", id="m1", additional_kwargs={"beam_chat_id": 7}),
        AIMessage(content="250ms.", id="m2", additional_kwargs={}),
    ]


def test_snapshot_round_trip_restores_memory_and_working_tail(tmp_path):
    memory = _memory()
    path = tmp_path / "snapshot.json"
    write_memory_snapshot(
        path,
        memory=memory,
        active_messages=_messages(),
        memory_profile="practical",
        run_id="run-1",
        source_commit="abc123",
        chat="BEAM/chats/100K/1/chat.json",
    )

    payload = load_memory_snapshot(path)
    restored = Memory(PRACTICAL_SECTIONS, policy=get_memory_policy("practical"))
    messages = restore_from_snapshot(payload, memory=restored, expected_profile="practical")

    assert restored.render() == memory.render()
    assert payload["run_id"] == "run-1" and payload["source_commit"] == "abc123"
    assert isinstance(messages[0], HumanMessage) and isinstance(messages[1], AIMessage)
    assert messages[0].content == "What is the latency?"
    assert messages[0].additional_kwargs["beam_chat_id"] == 7
    assert messages[0].id == "m1"


def test_restore_rejects_memory_profile_mismatch(tmp_path):
    path = tmp_path / "snapshot.json"
    write_memory_snapshot(
        path,
        memory=_memory(),
        active_messages=[],
        memory_profile="eval",
        run_id="run-1",
        source_commit=None,
    )
    restored = Memory(PRACTICAL_SECTIONS, policy=get_memory_policy("practical"))

    with pytest.raises(ValueError, match="does not match"):
        restore_from_snapshot(
            load_memory_snapshot(path), memory=restored, expected_profile="practical"
        )


def test_load_rejects_unsupported_snapshot_version(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"snapshot_version": 999}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported memory snapshot version"):
        load_memory_snapshot(path)

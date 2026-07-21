"""Frozen structured-memory snapshots for paired BEAM replay.

A snapshot freezes the post-ingestion memory state and working-context tail so
selector and answer experiments can be replayed against identical memory,
removing ingestion stochasticity from A/B comparisons. This is an evaluation
concern: production code never reads snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from memory_agent.core.store import Memory

SNAPSHOT_VERSION = 1


def serialize_messages(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user" if isinstance(message, HumanMessage) else "assistant",
            "content": message.content,
            "id": message.id,
            "additional_kwargs": dict(message.additional_kwargs),
        }
        for message in messages
    ]


def deserialize_messages(raw: list[dict[str, Any]]) -> list[AnyMessage]:
    messages: list[AnyMessage] = []
    for item in raw:
        message_cls = HumanMessage if item.get("role") == "user" else AIMessage
        messages.append(
            message_cls(
                content=str(item.get("content", "")),
                id=item.get("id"),
                additional_kwargs=dict(item.get("additional_kwargs") or {}),
            )
        )
    return messages


def write_memory_snapshot(
    path: Path,
    *,
    memory: Memory,
    active_messages: list[AnyMessage],
    run_id: str,
    source_commit: str | None,
    chat: str | None = None,
) -> None:
    payload = {
        "snapshot_version": SNAPSHOT_VERSION,
        "run_id": run_id,
        "source_commit": source_commit,
        "chat": chat,
        "memory_state": memory.to_state(),
        "active_messages": serialize_messages(active_messages),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_memory_snapshot(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("snapshot_version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported memory snapshot version: {version}")
    return payload


def restore_from_snapshot(
    payload: dict[str, Any],
    *,
    memory: Memory,
) -> list[AnyMessage]:
    """Load snapshot memory state and return its working-context messages."""
    memory.load_state(payload["memory_state"])
    return deserialize_messages(payload.get("active_messages", []))

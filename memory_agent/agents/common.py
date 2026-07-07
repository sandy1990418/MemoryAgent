"""Shared helpers for runnable LangChain agents."""

from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant. Use tools when they help. "
    "When a user asks for calculation or weather, call the relevant tool."
)


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def invoke_agent(agent: Any, prompt: str, thread_id: str) -> dict[str, Any]:
    return agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=thread_config(thread_id),
    )


def print_last_message(result: dict[str, Any]) -> None:
    last_message = result["messages"][-1]
    print(last_message.content)

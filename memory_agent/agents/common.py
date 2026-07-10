"""Shared helpers for runnable LangChain agents."""

from __future__ import annotations

from typing import Any

from memory_agent.clients.llm import LangChainTokenCallback, TokenLedger

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant. Use available tools when they help."
)


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def invoke_agent(
    agent: Any,
    prompt: str,
    thread_id: str,
    token_ledger: TokenLedger | None = None,
) -> dict[str, Any]:
    config = thread_config(thread_id)
    if token_ledger is not None:
        config["callbacks"] = [LangChainTokenCallback(token_ledger, "agent")]
    return agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )


def print_last_message(result: dict[str, Any]) -> None:
    last_message = result["messages"][-1]
    print(last_message.content)

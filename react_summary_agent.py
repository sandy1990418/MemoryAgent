"""Minimal LangChain agent with summarization middleware.

Run:
    export OPENAI_API_KEY="..."
    python react_summary_agent.py

Optional:
    export MAIN_MODEL="openai:gpt-5.5"
    export SUMMARY_MODEL="openai:gpt-5.4-mini"
"""

from __future__ import annotations

import ast
import operator
import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


MAIN_MODEL = os.getenv("MAIN_MODEL", "openai:gpt-5.5")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "openai:gpt-5.4-mini")
DEFAULT_THREAD_ID = "react-summary-demo"


@tool
def weather(city: str) -> str:
    """Return a tiny mock weather report for a city."""
    return f"{city}: sunny, 26 C, light wind. This is mock data for the demo."


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    allowed_binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    allowed_unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary_ops:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return allowed_binary_ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary_ops:
            return allowed_unary_ops[type(node.op)](eval_node(node.operand))
        raise ValueError("Only basic arithmetic is supported.")

    parsed = ast.parse(expression, mode="eval")
    return str(eval_node(parsed))


def build_agent(
    main_model: str | None = None,
    summary_model: str | None = None,
    checkpointer: InMemorySaver | None = None,
):
    """Build a ReAct-style LangChain agent with conversation summarization."""
    return create_agent(
        model=main_model or MAIN_MODEL,
        tools=[weather, calculator],
        system_prompt=(
            "You are a concise assistant. Use tools when they help. "
            "When a user asks for calculation or weather, call the relevant tool."
        ),
        checkpointer=checkpointer or InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model=summary_model or SUMMARY_MODEL,
                # Latest LangChain uses trigger/keep instead of the deprecated
                # max_tokens_before_summary/messages_to_keep arguments.
                trigger=("messages", 6),
                keep=("messages", 2),
            ),
        ],
    )


def thread_config(thread_id: str = DEFAULT_THREAD_ID) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def invoke_agent(agent, prompt: str, thread_id: str = DEFAULT_THREAD_ID) -> dict[str, Any]:
    return agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=thread_config(thread_id),
    )


def print_last_message(result: dict[str, Any]) -> None:
    last_message = result["messages"][-1]
    print(last_message.content)


def main() -> None:
    agent = build_agent()

    prompts = [
        "Hi, my name is Hannah. Please remember I am testing summary middleware.",
        "What is 18 * 23 + 7? Use the calculator.",
        "What is the mock weather in Taipei? Use the weather tool.",
        "I also like concise bullet points. Please remember that.",
        "What did I tell you about myself earlier?",
    ]

    for prompt in prompts:
        print(f"\nUser: {prompt}")
        result = invoke_agent(agent, prompt)
        print("Agent:", end=" ")
        print_last_message(result)


if __name__ == "__main__":
    main()

"""LangChain ReAct agent demo using structured memory instead of summarization.

Run:
    export OPENAI_API_KEY="..."
    python demo_react.py

Optional:
    export MAIN_MODEL="openai:gpt-5.5"
    export SUMMARY_MODEL="gpt-5.4-mini"

Reads OPENAI_API_KEY from the environment or from a .env file next to this
script (same convention as demo.py).
"""

from __future__ import annotations

import ast
import operator
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from memory_agent import AGENT_SECTIONS, Memory, MemoryUpdater, OpenAIClient
from memory_agent.langchain_middleware import StructuredMemoryMiddleware

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MAIN_MODEL = os.getenv("MAIN_MODEL", "openai:gpt-5.5")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-5.4-mini")
DEFAULT_THREAD_ID = "react-structured-memory-demo"


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


PROMPTS = [
    "Hi, I'm Hannah. DECISION: we will use in-memory storage for this project's cache layer.",
    "What's the mock weather like in Taipei? Use the weather tool.",
    "What is 42 * 17 + 5? Use the calculator.",
    "I prefer bullet points for your answers, please remember that.",
    "Let's chat for a moment: what's a good name for a cat?",
    "Any general tips for good code review practices?",
    "Actually, I've changed my mind: I prefer detailed paragraphs instead of bullet points.",
    "One more unrelated question: recommend a good weekend hobby.",
    "Let's continue the code review discussion, any more tips?",
    (
        "Can you remind me: what did we decide early on about storage, what is my "
        "current preference for answer style, and what did the calculator return earlier?"
    ),
]


def build_agent():
    """Build a ReAct-style LangChain agent backed by structured living memory."""
    memory = Memory(sections=AGENT_SECTIONS)
    updater = MemoryUpdater(llm=OpenAIClient(SUMMARY_MODEL), sections=AGENT_SECTIONS)
    middleware = StructuredMemoryMiddleware(
        memory=memory,
        updater=updater,
        max_tokens=600,
    )

    agent = create_agent(
        model=MAIN_MODEL,
        tools=[weather, calculator],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    return agent, middleware


def thread_config(thread_id: str = DEFAULT_THREAD_ID) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def invoke_agent(agent, prompt: str, thread_id: str = DEFAULT_THREAD_ID) -> dict:
    return agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=thread_config(thread_id),
    )


def print_last_message(result: dict) -> None:
    last_message = result["messages"][-1]
    print(last_message.content)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set (checked environment and .env). "
            "This demo calls the real OpenAI API, so it needs a key to run.\n"
            "Skipping demo run."
        )
        return

    agent, middleware = build_agent()

    for prompt in PROMPTS:
        print(f"\nUser: {prompt}")
        result = invoke_agent(agent, prompt)
        print("Agent:", end=" ")
        print_last_message(result)

    print("\n\n=== Final memory (including superseded entries) ===\n")
    print(middleware.memory.render(include_superseded=True))
    print(f"\nTranscript length: {len(middleware.transcript)}")


if __name__ == "__main__":
    main()

"""LangChain ReAct agent with structured memory plus mem0 long-term vector memory.

Run:
    export OPENAI_API_KEY="..."
    python react_hybrid_agent.py

Optional:
    export MAIN_MODEL="openai:gpt-5.5"
    export MEMORY_MODEL="openai:gpt-5.4-mini"
    export STRUCTURED_MAX_TOKENS="220"
    export MEM0_LLM_MODEL="gpt-4o-mini"
    export MEM0_USER_ID="demo-user"
    export MEM0_DATA_DIR=".mem0"

Reads OPENAI_API_KEY from the environment or from a .env file next to this
script. If mem0ai is not installed, the demo falls back to structured memory
only.
"""

from __future__ import annotations

import ast
import operator
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from memory_agent import AGENT_SECTIONS, Mem0LongTermMemory, Memory, MemoryUpdater, OpenAIClient
from memory_agent.langchain_middleware import StructuredMemoryMiddleware
from memory_agent.longterm_middleware import LongTermMemoryMiddleware

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
os.environ.setdefault("MEM0_TELEMETRY", "False")

MAIN_MODEL = os.getenv("MAIN_MODEL", "openai:gpt-5.4-nano")
MEMORY_MODEL = os.getenv("MEMORY_MODEL", os.getenv("SUMMARY_MODEL", "openai:gpt-5.4-nano"))
STRUCTURED_MAX_TOKENS = int(os.getenv("STRUCTURED_MAX_TOKENS", "220"))
STRUCTURED_MAX_MEMORY_TOKENS = int(os.getenv("STRUCTURED_MAX_MEMORY_TOKENS", "600"))
MEM0_USER_ID = os.getenv("MEM0_USER_ID", "demo-user")
MEM0_DATA_DIR = os.getenv("MEM0_DATA_DIR", ".mem0")
MEM0_LLM_MODEL = os.getenv("MEM0_LLM_MODEL", "gpt-5.4-nano")
os.environ.setdefault("MEM0_DIR", MEM0_DATA_DIR)
DEFAULT_THREAD_ID = "react-hybrid-memory-demo"


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
    (
        "Hi, my name is Hannah. Please remember that my favorite city is Taipei "
        "and my project codename is Lantern."
    ),
    "What is 18 * 23 + 7? Use the calculator.",
    "What is the mock weather in Taipei? Use the weather tool.",
    "Please remember that I prefer concise answers with concrete next steps.",
    "Let's pad the thread a bit: suggest one harmless status-check question for a teammate.",
    "Another filler turn: name two tradeoffs of summarizing conversation context.",
    "One more filler turn: what should a small demo log after each model call?",
    (
        "Can you recall the early facts I gave you about my name, favorite city, "
        "project codename, and answer-style preference?"
    ),
]


def _structured_middleware(memory_model: str) -> StructuredMemoryMiddleware:
    memory = Memory(sections=AGENT_SECTIONS)
    updater = MemoryUpdater(llm=OpenAIClient(memory_model), sections=AGENT_SECTIONS)
    return StructuredMemoryMiddleware(
        memory=memory,
        updater=updater,
        max_tokens=STRUCTURED_MAX_TOKENS,
        max_memory_tokens=STRUCTURED_MAX_MEMORY_TOKENS,
    )


def build_agent(
    main_model: str | None = None,
    summary_model: str | None = None,
    memory_model: str | None = None,
    user_id: str | None = None,
    data_dir: str | None = None,
    checkpointer: InMemorySaver | None = None,
) -> tuple[Any, StructuredMemoryMiddleware, LongTermMemoryMiddleware | None]:
    """Build a ReAct-style agent with structured compression and optional mem0 recall."""
    resolved_memory_model = memory_model or summary_model or MEMORY_MODEL
    structured_middleware = _structured_middleware(resolved_memory_model)
    middleware: list[Any] = [structured_middleware]
    long_term_middleware: LongTermMemoryMiddleware | None = None

    try:
        long_term = Mem0LongTermMemory.from_local(
            data_dir=data_dir or MEM0_DATA_DIR,
            llm_model=MEM0_LLM_MODEL,
        )
    except ImportError:
        print(
            "mem0ai is not installed; install it with `pip install mem0ai>=2.0` "
            "to enable long-term vector memory. Running with structured memory only."
        )
    else:
        long_term_middleware = LongTermMemoryMiddleware(
            long_term=long_term,
            user_id=user_id or MEM0_USER_ID,
        )
        middleware.append(long_term_middleware)

    agent = create_agent(
        model=main_model or MAIN_MODEL,
        tools=[weather, calculator],
        system_prompt=(
            "You are a concise assistant. Use tools when they help. "
            "When a user asks for calculation or weather, call the relevant tool. "
            "If injected # Conversation Memory and # Long-Term Memory conflict, "
            "prefer # Conversation Memory as the current structured state and use "
            "# Long-Term Memory as supporting raw recall."
        ),
        checkpointer=checkpointer or InMemorySaver(),
        middleware=middleware,
    )
    return agent, structured_middleware, long_term_middleware


def thread_config(thread_id: str = DEFAULT_THREAD_ID) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def invoke_agent(agent: Any, prompt: str, thread_id: str = DEFAULT_THREAD_ID) -> dict[str, Any]:
    return agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=thread_config(thread_id),
    )


def print_last_message(result: dict[str, Any]) -> None:
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

    agent, structured_middleware, long_term_middleware = build_agent()

    for prompt in PROMPTS:
        print(f"\nUser: {prompt}")
        result = invoke_agent(agent, prompt)
        print("Agent:", end=" ")
        print_last_message(result)
        if long_term_middleware is not None:
            for hit in long_term_middleware.last_recalled:
                print(f"[long-term recall] {hit.text}")

    persisted = long_term_middleware.flush() if long_term_middleware is not None else 0
    print(f"\nPersisted {persisted} message(s) to long-term memory at session end.")
    if long_term_middleware is None:
        print("Long-term vector memory was disabled because mem0ai is not installed.")
    else:
        print("Re-run this script to see cross-session semantic recall from .mem0/.")

    print("\n=== Structured memory (including superseded entries) ===\n")
    print(structured_middleware.memory.render(include_superseded=True) or "(No structured memory entries.)")
    print(f"\nTranscript length: {len(structured_middleware.transcript)}")


if __name__ == "__main__":
    main()

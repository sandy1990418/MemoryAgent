"""LangChain agent builders used by runnable scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from memory_agent.config import HybridAgentConfig, StructuredAgentConfig, SummaryAgentConfig
from memory_agent.demo_tools import DEMO_TOOLS
from memory_agent.langchain_middleware import StructuredMemoryMiddleware
from memory_agent.longterm_middleware import LongTermMemoryMiddleware
from memory_agent.longterm import Mem0LongTermMemory
from memory_agent.llm import OpenAIClient
from memory_agent.memory import Memory
from memory_agent.sections import AGENT_SECTIONS
from memory_agent.updater import MemoryUpdater


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant. Use tools when they help. "
    "When a user asks for calculation or weather, call the relevant tool."
)


@dataclass(frozen=True)
class StructuredAgentRuntime:
    agent: Any
    structured_middleware: StructuredMemoryMiddleware


@dataclass(frozen=True)
class HybridAgentRuntime:
    agent: Any
    structured_middleware: StructuredMemoryMiddleware
    long_term_middleware: LongTermMemoryMiddleware | None


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


def build_summary_agent(
    config: SummaryAgentConfig,
    checkpointer: InMemorySaver | None = None,
) -> Any:
    return create_agent(
        model=config.main_model,
        tools=DEMO_TOOLS,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model=config.summary_model,
                trigger=("messages", 6),
                keep=("messages", 2),
            ),
        ],
    )


def build_structured_middleware(config: StructuredAgentConfig) -> StructuredMemoryMiddleware:
    memory = Memory(sections=AGENT_SECTIONS)
    updater = MemoryUpdater(llm=OpenAIClient(config.memory_model), sections=AGENT_SECTIONS)
    return StructuredMemoryMiddleware(
        memory=memory,
        updater=updater,
        max_tokens=config.max_tokens,
        max_memory_tokens=config.max_memory_tokens,
    )


def build_structured_agent(
    config: StructuredAgentConfig,
    checkpointer: InMemorySaver | None = None,
) -> StructuredAgentRuntime:
    structured_middleware = build_structured_middleware(config)
    agent = create_agent(
        model=config.main_model,
        tools=DEMO_TOOLS,
        middleware=[structured_middleware],
        checkpointer=checkpointer or InMemorySaver(),
    )
    return StructuredAgentRuntime(agent=agent, structured_middleware=structured_middleware)


def _structured_config_from_hybrid(config: HybridAgentConfig) -> StructuredAgentConfig:
    return StructuredAgentConfig(
        main_model=config.main_model,
        memory_model=config.memory_model,
        thread_id=config.thread_id,
        max_tokens=config.structured_max_tokens,
        max_memory_tokens=config.structured_max_memory_tokens,
    )


def build_hybrid_agent(
    config: HybridAgentConfig,
    checkpointer: InMemorySaver | None = None,
) -> HybridAgentRuntime:
    structured_middleware = build_structured_middleware(_structured_config_from_hybrid(config))
    middleware: list[Any] = [structured_middleware]
    long_term_middleware: LongTermMemoryMiddleware | None = None

    os.environ.setdefault("MEM0_DIR", config.mem0_data_dir)
    try:
        long_term = Mem0LongTermMemory.from_local(
            data_dir=config.mem0_data_dir,
            llm_model=config.mem0_llm_model,
        )
    except ImportError:
        print(
            "mem0ai is not installed; install it with `pip install mem0ai>=2.0` "
            "to enable long-term vector memory. Running with structured memory only."
        )
    else:
        long_term_middleware = LongTermMemoryMiddleware(
            long_term=long_term,
            user_id=config.mem0_user_id,
        )
        middleware.append(long_term_middleware)

    agent = create_agent(
        model=config.main_model,
        tools=DEMO_TOOLS,
        system_prompt=(
            DEFAULT_SYSTEM_PROMPT
            + " If injected # Conversation Memory and # Long-Term Memory conflict, "
            "prefer # Conversation Memory as the current structured state and use "
            "# Long-Term Memory as supporting raw recall."
        ),
        checkpointer=checkpointer or InMemorySaver(),
        middleware=middleware,
    )
    return HybridAgentRuntime(
        agent=agent,
        structured_middleware=structured_middleware,
        long_term_middleware=long_term_middleware,
    )

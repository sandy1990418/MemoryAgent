"""Structured-memory plus long-term mem0 LangChain agent assembly."""

from __future__ import annotations

import os
from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from memory_agent.agents.common import DEFAULT_SYSTEM_PROMPT
from memory_agent.agents.structured import build_structured_middleware
from memory_agent.clients.mem0 import LongTermMemory, Mem0LongTermMemory
from memory_agent.longterm.middleware import LongTermMemoryMiddleware
from memory_agent.models.config import HybridAgentConfig, StructuredAgentConfig
from memory_agent.models.runtime import HybridAgentRuntime
from memory_agent.tools.demo import DEMO_TOOLS


def _structured_config_from_hybrid(config: HybridAgentConfig) -> StructuredAgentConfig:
    return StructuredAgentConfig(
        main_model=config.main_model,
        memory_model=config.memory_model,
        thread_id=config.thread_id,
        max_tokens=config.structured_max_tokens,
        max_memory_tokens=config.structured_max_memory_tokens,
        keep_messages=config.structured_keep_messages,
        memory_profile=config.memory_profile,
    )


def build_long_term_memory(config: HybridAgentConfig) -> LongTermMemory | None:
    """Build the long-term memory backend selected by HybridAgentConfig."""
    if config.mem0_backend == "disabled":
        return None

    if config.mem0_backend == "platform":
        return Mem0LongTermMemory.from_platform(api_key=config.mem0_api_key)

    if config.mem0_backend == "local":
        data_dir = config.mem0_data_dir or ".mem0"
        os.environ.setdefault("MEM0_DIR", data_dir)
        return Mem0LongTermMemory.from_local(
            data_dir=data_dir,
            llm_model=config.mem0_llm_model,
        )

    raise ValueError(f"Unsupported mem0 backend: {config.mem0_backend}")


def build_hybrid_agent(
    config: HybridAgentConfig,
    checkpointer: InMemorySaver | None = None,
    long_term_memory: LongTermMemory | None = None,
) -> HybridAgentRuntime:
    structured_middleware = build_structured_middleware(_structured_config_from_hybrid(config))
    middleware: list[Any] = [structured_middleware]
    long_term_middleware: LongTermMemoryMiddleware | None = None

    try:
        long_term = long_term_memory if long_term_memory is not None else build_long_term_memory(config)
    except ImportError:
        print(
            "mem0ai is not installed; install it with `pip install mem0ai>=2.0` "
            "to enable long-term vector memory. Running with structured memory only."
        )
    else:
        if long_term is not None:
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

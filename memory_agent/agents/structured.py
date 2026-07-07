"""Structured-memory LangChain agent assembly."""

from __future__ import annotations

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from memory_agent.clients.llm import OpenAIClient
from memory_agent.models.config import StructuredAgentConfig
from memory_agent.models.runtime import StructuredAgentRuntime
from memory_agent.models.sections import AGENT_SECTIONS
from memory_agent.structured.memory import Memory
from memory_agent.structured.middleware import StructuredMemoryMiddleware
from memory_agent.structured.updater import MemoryUpdater
from memory_agent.tools.demo import DEMO_TOOLS


def build_structured_middleware(config: StructuredAgentConfig) -> StructuredMemoryMiddleware:
    memory = Memory(sections=AGENT_SECTIONS)
    updater = MemoryUpdater(llm=OpenAIClient(config.memory_model), sections=AGENT_SECTIONS)
    return StructuredMemoryMiddleware(
        memory=memory,
        updater=updater,
        max_tokens=config.max_tokens,
        keep_messages=config.keep_messages,
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

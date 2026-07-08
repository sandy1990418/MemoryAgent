"""Structured-memory LangChain agent assembly."""

from __future__ import annotations

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from memory_agent.clients.llm import OpenAIClient
from memory_agent.models.config import StructuredAgentConfig
from memory_agent.models.policy import get_memory_policy
from memory_agent.models.runtime import StructuredAgentRuntime
from memory_agent.models.sections import sections_for_preset
from memory_agent.structured.memory import Memory
from memory_agent.structured.middleware import StructuredMemoryMiddleware
from memory_agent.structured.updater import MemoryUpdater
from memory_agent.tools.demo import DEMO_TOOLS


def build_structured_middleware(config: StructuredAgentConfig) -> StructuredMemoryMiddleware:
    policy = get_memory_policy(config.memory_profile)
    sections = sections_for_preset(policy.section_preset)
    memory = Memory(sections=sections, policy=policy)
    updater = MemoryUpdater(
        llm=OpenAIClient(config.memory_model),
        sections=sections,
        policy=policy,
    )
    return StructuredMemoryMiddleware(
        memory=memory,
        updater=updater,
        policy=policy,
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

"""Structured-memory LangChain agent assembly."""

from __future__ import annotations

from typing import Any, Sequence

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from memory_agent.clients.llm import OpenAIClient, TokenLedger
from memory_agent.models.config import StructuredAgentConfig
from memory_agent.models.policy import get_memory_policy, is_chat_policy
from memory_agent.models.runtime import StructuredAgentRuntime
from memory_agent.models.sections import sections_for_preset
from memory_agent.structured.compactor import MemoryCompactor
from memory_agent.structured.memory import Memory
from memory_agent.structured.middleware import StructuredMemoryMiddleware
from memory_agent.structured.updater import MemoryUpdater


def build_structured_middleware(
    config: StructuredAgentConfig,
    token_ledger: TokenLedger | None = None,
) -> StructuredMemoryMiddleware:
    ledger = token_ledger or TokenLedger()
    ledger.ensure_roles("updater", "compactor", "agent", "judge")
    policy = get_memory_policy(config.memory_profile)
    sections = sections_for_preset(config.memory_sections or policy.section_preset)
    memory = Memory(sections=sections, policy=policy)
    updater = MemoryUpdater(
        llm=OpenAIClient(
            config.memory_model,
            role="updater",
            token_ledger=ledger,
        ),
        sections=sections,
        policy=policy,
    )
    # Practical memory stays small: consolidate same-subject entries once the
    # active set grows. Eval/agent profiles keep granular entries on purpose.
    compactor = (
        MemoryCompactor(
            llm=OpenAIClient(
                config.memory_model,
                role="compactor",
                token_ledger=ledger,
            ),
            sections=sections,
            policy=policy,
        )
        if is_chat_policy(policy)
        else None
    )
    middleware = StructuredMemoryMiddleware(
        memory=memory,
        updater=updater,
        policy=policy,
        max_tokens=config.max_tokens,
        keep_messages=config.keep_messages,
        max_memory_tokens=config.max_memory_tokens,
        compactor=compactor,
        compact_min_active_entries=config.compact_min_active_entries,
    )
    middleware.token_ledger = ledger
    return middleware


def build_structured_agent(
    config: StructuredAgentConfig,
    checkpointer: InMemorySaver | None = None,
    tools: Sequence[Any] = (),
) -> StructuredAgentRuntime:
    token_ledger = TokenLedger()
    structured_middleware = build_structured_middleware(config, token_ledger)
    agent = create_agent(
        model=config.main_model,
        tools=list(tools),
        middleware=[structured_middleware],
        checkpointer=checkpointer or InMemorySaver(),
    )
    return StructuredAgentRuntime(
        agent=agent,
        structured_middleware=structured_middleware,
        token_ledger=token_ledger,
    )

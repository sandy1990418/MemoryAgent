"""Summary-middleware baseline agent assembly."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from memory_agent.agents.common import DEFAULT_SYSTEM_PROMPT
from memory_agent.models.config import SummaryAgentConfig
from memory_agent.tools.demo import DEMO_TOOLS


def build_summary_agent(
    config: SummaryAgentConfig,
    checkpointer: InMemorySaver | None = None,
) -> Any:
    """Build the baseline LangChain agent that uses built-in summarization."""
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

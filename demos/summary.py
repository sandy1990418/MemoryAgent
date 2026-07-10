"""Summary-middleware baseline used only for comparison demos."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from demos.config import SummaryAgentConfig
from demos.tools import DEMO_TOOLS
from memory_agent.agents.common import DEFAULT_SYSTEM_PROMPT


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


"""LangChain agent demo using structured memory."""

from __future__ import annotations

import json
import os

from demos.tools import DEMO_TOOLS
from memory_agent.agents import build_structured_agent, invoke_agent, print_last_message
from memory_agent.models.config import StructuredAgentConfig, load_project_env

PROMPTS = [
    "Hi, I'm Hannah. DECISION: use in-memory storage for this project's cache layer.",
    "What's the mock weather like in Taipei? Use the weather tool.",
    "What is 42 * 17 + 5? Use the calculator.",
    "I prefer bullet points for your answers, please remember that.",
    "Actually, I prefer detailed paragraphs instead of bullet points.",
    "Remind me what we decided, my current answer style, and the calculator result.",
]


def main() -> None:
    load_project_env()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; skipping the API-backed demo.")
        return

    config = StructuredAgentConfig.from_yaml_env()
    runtime = build_structured_agent(config, tools=DEMO_TOOLS)
    for prompt in PROMPTS:
        print(f"\nUser: {prompt}")
        result = invoke_agent(
            runtime.agent,
            prompt,
            thread_id=config.thread_id,
            token_ledger=runtime.token_ledger,
        )
        print("Agent:", end=" ")
        print_last_message(result)
    print("\n\n=== Final memory (including superseded entries) ===\n")
    print(runtime.structured_middleware.memory.render(include_superseded=True))
    print("\nToken usage by role:")
    print(json.dumps(runtime.token_ledger.to_dict(), indent=2))


if __name__ == "__main__":
    main()


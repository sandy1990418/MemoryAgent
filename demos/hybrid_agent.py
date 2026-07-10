"""LangChain agent demo using structured memory plus mem0 recall."""

from __future__ import annotations

import os

from demos.tools import DEMO_TOOLS
from memory_agent.agents import build_hybrid_agent, invoke_agent, print_last_message
from memory_agent.models.config import HybridAgentConfig, load_project_env

PROMPTS = [
    "My favorite city is Taipei and my project codename is Lantern.",
    "What is 18 * 23 + 7? Use the calculator.",
    "What is the mock weather in Taipei? Use the weather tool.",
    "Please remember that I prefer concise answers with concrete next steps.",
    "Recall my favorite city, project codename, and answer-style preference.",
]


def main() -> None:
    load_project_env()
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; skipping the API-backed demo.")
        return

    config = HybridAgentConfig.from_env()
    runtime = build_hybrid_agent(config, tools=DEMO_TOOLS)
    for prompt in PROMPTS:
        print(f"\nUser: {prompt}")
        result = invoke_agent(runtime.agent, prompt, thread_id=config.thread_id)
        print("Agent:", end=" ")
        print_last_message(result)

    persisted = runtime.long_term_middleware.flush() if runtime.long_term_middleware else 0
    print(f"\nPersisted {persisted} message(s) to long-term memory.")
    print("\n=== Structured memory ===\n")
    print(
        runtime.structured_middleware.memory.render(include_superseded=True)
        or "(No structured memory entries.)"
    )


if __name__ == "__main__":
    main()


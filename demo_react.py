"""LangChain ReAct agent demo using structured memory."""

from __future__ import annotations

import os

from memory_agent.agents import build_structured_agent, invoke_agent, print_last_message
from memory_agent.models.config import StructuredAgentConfig, load_project_env


PROMPTS = [
    "Hi, I'm Hannah. DECISION: we will use in-memory storage for this project's cache layer.",
    "What's the mock weather like in Taipei? Use the weather tool.",
    "What is 42 * 17 + 5? Use the calculator.",
    "I prefer bullet points for your answers, please remember that.",
    "Let's chat for a moment: what's a good name for a cat?",
    "Any general tips for good code review practices?",
    "Actually, I've changed my mind: I prefer detailed paragraphs instead of bullet points.",
    "One more unrelated question: recommend a good weekend hobby.",
    "Let's continue the code review discussion, any more tips?",
    (
        "Can you remind me: what did we decide early on about storage, what is my "
        "current preference for answer style, and what did the calculator return earlier?"
    ),
]


def main() -> None:
    load_project_env()
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set (checked environment and .env). "
            "This demo calls the real OpenAI API, so it needs a key to run.\n"
            "Skipping demo run."
        )
        return

    config = StructuredAgentConfig.from_env()
    runtime = build_structured_agent(config)

    for prompt in PROMPTS:
        print(f"\nUser: {prompt}")
        result = invoke_agent(runtime.agent, prompt, thread_id=config.thread_id)
        print("Agent:", end=" ")
        print_last_message(result)

    print("\n\n=== Final memory (including superseded entries) ===\n")
    print(runtime.structured_middleware.memory.render(include_superseded=True))
    print(f"\nTranscript length: {len(runtime.structured_middleware.transcript)}")


if __name__ == "__main__":
    main()

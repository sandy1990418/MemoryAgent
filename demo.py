"""Runnable demo of the structured living-summary memory system.

Run:
    python demo.py

Reads OPENAI_API_KEY from the environment or from a .env file next to
this script.

Optional:
    export MAIN_MODEL="openai:gpt-5.5"
    export SUMMARY_MODEL="openai:gpt-5.4-mini"
"""

from __future__ import annotations

import os

from memory_agent import CHAT_SECTIONS, MemorySession, MemoryUpdater, OpenAIClient
from memory_agent.models.config import SessionDemoConfig, load_project_env

PROMPTS = [
    "Hi, my name is Hannah.",
    "DECISION: we will use in-memory storage for this project's cache layer.",
    "我喜歡簡潔的條列式回答，請記住這點。",
    "What's the mock weather like in general, just chat with me about small talk for a moment.",
    "Let's talk about something else: what are good practices for code review?",
    "I've changed my mind, I actually prefer detailed paragraph explanations instead of bullet points.",
    "One more unrelated question: what is a good name for a cat?",
    "Let's continue discussing code review practices, any more tips?",
    "CHANGE: we now use file storage instead of in-memory storage, due to durability concerns.",
    "Can you remind me what we decided about storage early in this conversation, and what my current preference for answer style is?",
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

    config = SessionDemoConfig.from_env()
    chat_llm = OpenAIClient(model=config.main_model)
    summary_llm = OpenAIClient(model=config.memory_model)
    updater = MemoryUpdater(llm=summary_llm, sections=CHAT_SECTIONS)

    session = MemorySession(
        chat_llm=chat_llm,
        updater=updater,
        sections=CHAT_SECTIONS,
        max_window_tokens=config.max_window_tokens,
        base_system_prompt="You are a helpful assistant.",
    )

    for prompt in PROMPTS:
        print(f"\nUser: {prompt}")
        reply = session.send(prompt)
        print(f"Assistant: {reply}")

    print("\n\n=== Final memory (including superseded entries) ===\n")
    print(session.memory.render(include_superseded=True))


if __name__ == "__main__":
    main()

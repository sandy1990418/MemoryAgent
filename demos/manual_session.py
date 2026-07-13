"""Legacy manual-session memory demo retained for comparison."""

from __future__ import annotations

import os

from demos.config import SessionDemoConfig
from memory_agent.clients.llm import OpenAIClient
from memory_agent.models.config import load_project_env
from memory_agent.application.session import MemorySession
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.update.updater import MemoryUpdater

PROMPTS = [
    "Hi, my name is Hannah.",
    "DECISION: we will use in-memory storage for this project's cache layer.",
    "我喜歡簡潔的條列式回答，請記住這點。",
    "What's the mock weather like in general, just chat with me about small talk for a moment.",
    "Let's talk about something else: what are good practices for code review?",
    "I've changed my mind, I actually prefer detailed paragraph explanations instead of bullet points.",
    "CHANGE: we now use file storage instead of in-memory storage, due to durability concerns.",
    "Can you remind me what we decided about storage and my current answer-style preference?",
]


def main() -> None:
    load_project_env()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; skipping the API-backed demo.")
        return

    config = SessionDemoConfig.from_env()
    session = MemorySession(
        chat_llm=OpenAIClient(model=config.main_model),
        updater=MemoryUpdater(
            llm=OpenAIClient(model=config.memory_model),
            sections=CHAT_SECTIONS,
        ),
        sections=CHAT_SECTIONS,
        max_window_tokens=config.max_window_tokens,
        base_system_prompt="You are a helpful assistant.",
    )
    for prompt in PROMPTS:
        print(f"\nUser: {prompt}")
        print(f"Assistant: {session.send(prompt)}")
    print("\n\n=== Final memory (including superseded entries) ===\n")
    print(session.memory.render(include_superseded=True))


if __name__ == "__main__":
    main()

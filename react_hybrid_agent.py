"""LangChain ReAct agent with structured memory plus mem0 long-term recall."""

from __future__ import annotations

import os

from memory_agent.agent_builders import build_hybrid_agent, invoke_agent, print_last_message
from memory_agent.config import HybridAgentConfig, load_project_env


PROMPTS = [
    (
        "Hi, my name is Hannah. Please remember that my favorite city is Taipei "
        "and my project codename is Lantern."
    ),
    "What is 18 * 23 + 7? Use the calculator.",
    "What is the mock weather in Taipei? Use the weather tool.",
    "Please remember that I prefer concise answers with concrete next steps.",
    "Let's pad the thread a bit: suggest one harmless status-check question for a teammate.",
    "Another filler turn: name two tradeoffs of summarizing conversation context.",
    "One more filler turn: what should a small demo log after each model call?",
    (
        "Can you recall the early facts I gave you about my name, favorite city, "
        "project codename, and answer-style preference?"
    ),
]


def main() -> None:
    load_project_env()
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set (checked environment and .env). "
            "This demo calls the real OpenAI API, so it needs a key to run.\n"
            "Skipping demo run."
        )
        return

    config = HybridAgentConfig.from_env()
    runtime = build_hybrid_agent(config)

    for prompt in PROMPTS:
        print(f"\nUser: {prompt}")
        result = invoke_agent(runtime.agent, prompt, thread_id=config.thread_id)
        print("Agent:", end=" ")
        print_last_message(result)
        if runtime.long_term_middleware is not None:
            for hit in runtime.long_term_middleware.last_recalled:
                print(f"[long-term recall] {hit.text}")

    persisted = (
        runtime.long_term_middleware.flush()
        if runtime.long_term_middleware is not None
        else 0
    )
    print(f"\nPersisted {persisted} message(s) to long-term memory at session end.")
    if runtime.long_term_middleware is None:
        print("Long-term vector memory was disabled because mem0ai is not installed.")
    else:
        print(f"Re-run this script to see cross-session semantic recall from {config.mem0_data_dir}/.")

    print("\n=== Structured memory (including superseded entries) ===\n")
    print(
        runtime.structured_middleware.memory.render(include_superseded=True)
        or "(No structured memory entries.)"
    )
    print(f"\nTranscript length: {len(runtime.structured_middleware.transcript)}")


if __name__ == "__main__":
    main()

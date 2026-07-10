"""Minimal comparison demo using LangChain SummarizationMiddleware."""

from demos.config import SummaryAgentConfig
from demos.summary import build_summary_agent
from memory_agent.agents import invoke_agent, print_last_message
from memory_agent.models.config import load_project_env

PROMPTS = [
    "Hi, my name is Hannah. Please remember I am testing summary middleware.",
    "What is 18 * 23 + 7? Use the calculator.",
    "What is the mock weather in Taipei? Use the weather tool.",
    "I also like concise bullet points. Please remember that.",
    "What did I tell you about myself earlier?",
]


def main() -> None:
    load_project_env()
    config = SummaryAgentConfig.from_env()
    agent = build_summary_agent(config)
    for prompt in PROMPTS:
        print(f"\nUser: {prompt}")
        result = invoke_agent(agent, prompt, thread_id=config.thread_id)
        print("Agent:", end=" ")
        print_last_message(result)


if __name__ == "__main__":
    main()


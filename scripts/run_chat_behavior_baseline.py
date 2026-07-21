"""Run a small production-like chat-memory behavior baseline.

This is evaluation tooling.  It calls only the public chat API and keeps
case-specific lexical expectations out of the production runtime.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from memory_agent import Turn, build_chat_memory

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class Scenario:
    name: str
    batches: tuple[tuple[Turn, ...], ...]
    check: Callable[[str], bool]


class FailingLLM:
    def complete(self, system, messages, model=None):
        raise RuntimeError("injected transport failure")


def scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            "english_preference",
            ((Turn(1, "user", "Please remember that I prefer concise answers with bullet points."),),),
            lambda text: "concise" in text.lower() and "bullet" in text.lower(),
        ),
        Scenario(
            "chinese_preference",
            ((Turn(1, "user", "請記住：我偏好使用繁體中文回答，而且先講結論。"),),),
            lambda text: "繁體中文" in text and "結論" in text,
        ),
        Scenario(
            "contradictory_project_state",
            (
                (Turn(1, "user", "The project currently uses SQLite for production data."),),
                (Turn(2, "user", "Correction: production now uses PostgreSQL, not SQLite."),),
            ),
            lambda text: "postgresql" in text.lower()
            and (
                "sqlite" not in text.lower()
                or "not sqlite" in text.lower()
                or "correct" in text.lower()
            ),
        ),
        Scenario(
            "ordinary_question",
            ((Turn(1, "user", "What is the capital of France?"), Turn(2, "assistant", "Paris.")),),
            lambda text: not text.strip(),
        ),
    )


def run() -> dict:
    results = []
    for scenario in scenarios():
        memory = build_chat_memory(compact=False)
        operations = []
        rejected = []
        for batch in scenario.batches:
            applied, failed = memory.update(list(batch))
            operations.extend(applied)
            rejected.extend(failed)
        rendered = memory.render()
        results.append({
            "name": scenario.name,
            "correct": scenario.check(rendered) and not rejected,
            "operations": operations,
            "rejected": rejected,
            "active_entries": sum(
                entry.status == "active" for entry in memory.memory.entries.values()
            ),
            "memory_length": len(rendered),
            "memory": rendered,
            "token_usage": memory.token_usage(),
        })

    failure_memory = build_chat_memory(llm=FailingLLM(), compact=False)
    before = failure_memory.memory.to_state()
    _applied, failure = failure_memory.update(
        [Turn(1, "user", "Remember that the launch is blocked.")]
    )
    failure_case = {
        "name": "updater_failure_atomicity",
        "correct": before == failure_memory.memory.to_state() and bool(failure),
        "failure": failure,
        "memory_unchanged": before == failure_memory.memory.to_state(),
        "token_usage": failure_memory.token_usage(),
    }

    updater_tokens = sum(
        result["token_usage"].get("updater", {}).get("total_tokens", 0)
        for result in results
    )
    return {
        "summary": {
            "cases": len(results),
            "correct": sum(result["correct"] for result in results),
            "active_entries": sum(result["active_entries"] for result in results),
            "memory_length": sum(result["memory_length"] for result in results),
            "updater_tokens": updater_tokens,
            "failure_case_correct": failure_case["correct"],
        },
        "cases": results,
        "failure_case": failure_case,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

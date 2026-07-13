"""Run production-style updater transaction and semantic-group diagnostics."""

from __future__ import annotations

import json
import time

from memory_agent.core.sections import AGENT_SECTIONS, PRACTICAL_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import get_memory_policy
from memory_agent.update.operations import UpdateFailed
from memory_agent.update.updater import MemoryUpdater


class RecordingLLM:
    def __init__(self, response: str = '[{"op":"NOOP"}]', fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.prompts: list[dict] = []

    def complete(self, system, messages, model=None):
        self.prompts.append({"system": system, "messages": messages})
        if self.fail:
            raise RuntimeError("injected transport failure")
        return self.response


def run_case(name: str, turns: list[Turn], *, fail: bool = False) -> dict:
    llm = RecordingLLM(fail=fail)
    updater = MemoryUpdater(
        llm=llm,
        sections=AGENT_SECTIONS,
        evicted_turn_token_budget=1200,
        update_memory_token_budget=600,
        max_candidate_entries=8,
        max_legacy_candidate_entries=4,
        enable_llm_gate=True,
    )
    memory = Memory(sections=AGENT_SECTIONS)
    before = memory.to_state()
    updater._turns_within_budget(turns)
    budget_selection = updater.turn_selection_reports[-1]
    started = time.perf_counter()
    error = None
    prepared = None
    try:
        prepared = updater.prepare_update(memory, turns)
        if not fail and not prepared.rejected_ops:
            prepared.commit(memory)
    except UpdateFailed as exc:
        error = str(exc)
    elapsed = time.perf_counter() - started
    prompt = llm.prompts[-1] if llm.prompts else None
    prompt_trace = None
    if prompt:
        text = prompt["system"] + "\n" + "\n".join(
            str(message.get("content", "")) for message in prompt["messages"]
        )
        prompt_trace = text[-2000:]
    return {
        "name": name,
        "turns": [turn.__dict__ for turn in turns],
        "selection": budget_selection,
        "prompt_tail": prompt_trace,
        "deterministic_and_llm_ops": prepared.applied_ops if prepared else [],
        "rejected_ops": prepared.rejected_ops if prepared else [],
        "trial_memory": prepared.trial_memory.to_state() if prepared else None,
        "live_before": before,
        "live_after": memory.to_state(),
        "live_unchanged": before == memory.to_state(),
        "error": error,
        "latency_seconds": round(elapsed, 6),
        "token_usage": updater.update_token_usage(),
    }


def run_lifecycle_case(name: str, updates: list[Turn]) -> dict:
    llm = RecordingLLM()
    updater = MemoryUpdater(
        llm=llm,
        sections=PRACTICAL_SECTIONS,
        policy=get_memory_policy("chat"),
        evicted_turn_token_budget=1200,
        update_memory_token_budget=600,
        max_candidate_entries=8,
        max_legacy_candidate_entries=4,
        enable_llm_gate=True,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS)
    states = []
    started = time.perf_counter()
    for turn in updates:
        before = memory.to_state()
        prepared = updater.prepare_update(memory, [turn])
        if not prepared.rejected_ops:
            prepared.commit(memory)
        states.append({
            "turn": turn.__dict__,
            "before": before,
            "trial": prepared.trial_memory.to_state(),
            "after": memory.to_state(),
            "applied_ops": prepared.applied_ops,
            "rejected_ops": prepared.rejected_ops,
        })
    return {
        "name": name,
        "updates": states,
        "final_memory": memory.to_state(),
        "latency_seconds": round(time.perf_counter() - started, 6),
        "token_usage": updater.update_token_usage(),
    }


def main() -> None:
    english = "The release must preserve every compatibility constraint. " * 120
    chinese = "這次更新必須保留所有相容性限制並完整驗證，不可以省略任何需求。" * 200
    cases = [
        run_case("oversized_english", [Turn(1, "user", english), Turn(2, "assistant", "I will preserve and verify all constraints.")]),
        run_case("oversized_chinese", [Turn(3, "user", chinese), Turn(4, "assistant", "我會保留並驗證全部限制。")]),
        run_case("acceptance", [Turn(5, "assistant", "I propose PostgreSQL."), Turn(6, "user", "Yes, go with that.")]),
        run_case("rejection", [Turn(7, "assistant", "I propose Redis."), Turn(8, "user", "No, reject that proposal.")]),
        run_case("correction", [Turn(9, "user", "The status is blocked."), Turn(10, "assistant", "Noted."), Turn(11, "user", "Actually, correction: the status is shipped.")]),
        run_case("status_changes", [Turn(12, "user", "The project is planned."), Turn(13, "assistant", "Noted."), Turn(14, "user", "Actually it is active."), Turn(15, "assistant", "Noted."), Turn(16, "user", "It is no longer active; it is complete.")]),
        run_case("single_tool_result", [Turn(17, "user", "Check status."), Turn(18, "assistant", "[tool_call] status({})"), Turn(19, "tool", "healthy")]),
        run_case("multiple_tool_results", [Turn(20, "user", "Check both."), Turn(21, "assistant", "[tool_call] checks({})"), Turn(22, "tool", "db healthy"), Turn(23, "tool", "api healthy")]),
        run_case("unresolved_user", [Turn(24, "user", "What is still blocked?")]),
        run_case("optional_older_mandatory_recent", [Turn(25, "user", "old " * 1500), Turn(26, "assistant", "old answer"), Turn(27, "user", "new mandatory request"), Turn(28, "assistant", "new answer")]),
        run_case("failure_retry", [Turn(29, "user", "Actually, I have never deployed it."), Turn(30, "assistant", "Understood.")], fail=True),
        run_lifecycle_case(
            "monthly_budget_lifecycle",
            [
                Turn(31, "user", "My monthly book budget is $35."),
                Turn(32, "user", "My monthly book budget increased to $50."),
                Turn(33, "user", "My monthly book budget is back to $35."),
            ],
        ),
        run_lifecycle_case(
            "distinct_goals_do_not_merge",
            [
                Turn(34, "user", "My emergency fund goal is $2,000."),
                Turn(35, "user", "My family car goal is $5,000."),
            ],
        ),
        run_case(
            "chinese_project_state_lifecycle",
            [
                Turn(36, "user", "專案是規劃中。"),
                Turn(37, "user", "它目前是進行中。"),
                Turn(38, "user", "它已經完成。"),
            ],
        ),
        run_case(
            "chinese_acceptance",
            [
                Turn(39, "assistant", "我建議採用 PostgreSQL。"),
                Turn(40, "user", "同意，就這樣。"),
            ],
        ),
        run_case(
            "durable_work_embedded_in_question",
            [
                Turn(
                    41,
                    "user",
                    "I'm trying to implement password hashing with Werkzeug.security, "
                    "but I'm not sure how to verify passwords correctly. Can you help?",
                ),
                Turn(42, "assistant", "Use generate_password_hash and check_password_hash."),
            ],
        ),
        run_case(
            "documentation_work_embedded_in_question",
            [
                Turn(
                    43,
                    "user",
                    "I'm working on project documentation in Confluence with API tables "
                    "and architecture diagrams. Can you review the structure?",
                ),
                Turn(44, "assistant", "Yes, organize it by endpoint and decision."),
            ],
        ),
        run_case(
            "generic_assistant_intro_is_not_a_proposal",
            [
                Turn(45, "assistant", "Certainly! Let's walk through the error."),
                Turn(46, "user", "No, that did not solve it."),
            ],
        ),
    ]
    print(
        json.dumps(
            cases,
            indent=2,
            ensure_ascii=False,
            default=lambda value: value.__dict__,
        )
    )


if __name__ == "__main__":
    main()

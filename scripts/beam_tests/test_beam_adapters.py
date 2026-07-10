import pytest

from evaluation.beam import BeamChatCaseAdapter, compare_fixed_budget_runs
from memory_agent.domain import EventSourceType


def test_beam_chat_adapter_outputs_generic_events_and_keeps_metadata_at_boundary():
    events = BeamChatCaseAdapter().adapt_messages([{"id": 7, "role": "user", "content": "hello"}], case_id="2")
    assert events[0].source_type == EventSourceType.CHAT_MESSAGE
    assert events[0].metadata["case_id"] == "2"


def test_fixed_budget_comparison_separates_production_and_judge_costs():
    rows = [
        {"variant": variant, "case_id": "1", "question_id": "q1", "context_budget_tokens": 256, "actual_context_tokens": 200, "quality": quality, "tokens": {"updater": 10, "compactor": 5, "agent": 20, "judge": 100}}
        for variant, quality in (("baseline", .5), ("current", .75))
    ]
    result = compare_fixed_budget_runs(rows)
    assert [item["variant"] for item in result] == ["baseline", "current"]
    assert result[0]["production_tokens"] == 35
    assert result[0]["judge_tokens"] == 100
    assert result[0]["validated_profile"] == "chat"
    assert result[0]["agent_memory_validated"] is False


def test_fixed_budget_comparison_rejects_mismatched_question_sets():
    with pytest.raises(ValueError, match="same cases"):
        compare_fixed_budget_runs([
            {"variant": "a", "case_id": "1", "question_id": "q1", "context_budget_tokens": 256},
            {"variant": "b", "case_id": "1", "question_id": "q2", "context_budget_tokens": 256},
        ])

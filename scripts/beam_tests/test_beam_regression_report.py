from evaluation.beam.regression_report import (
    aggregate_runs,
    attribute_question_failure,
    compare_aggregates,
)


def _item(*, context="", response="", score=0.0, heuristic=False, judge=False):
    return {
        "question": "What is the API latency?",
        "answer_context": context,
        "llm_response": response,
        "selected_memory_ids": ["F1"] if context else [],
        "llm_judge_score": score,
        "rubric_checks": [{"target": "API latency is 250ms", "hit": heuristic}],
        "judge_checks": [{"passed": judge}],
    }


def test_failure_attribution_distinguishes_ingestion_selection_and_answer():
    ingestion = attribute_question_failure(_item(), full_memory="unrelated", compactor_metrics=None)
    selection = attribute_question_failure(
        _item(), full_memory="API latency is 250ms", compactor_metrics=None
    )
    answer = attribute_question_failure(
        _item(context="API latency is 250ms", response="I do not know"),
        full_memory="API latency is 250ms", compactor_metrics=None,
    )
    assert ingestion["stage"] == "ingestion_update_failure"
    assert selection["stage"] == "selection_failure"
    assert answer["stage"] == "answer_generation_failure"


def test_heuristic_miss_with_partial_judge_credit_is_not_judge_failure():
    attribution = attribute_question_failure(
        _item(
            context="API latency is 250ms", response="The measured latency was about 250ms.",
            score=0.5, heuristic=False, judge=False,
        ),
        full_memory="API latency is 250ms", compactor_metrics=None,
    )
    assert attribution["stage"] == "answer_generation_failure"


def test_aggregate_and_comparison_include_quality_cost_and_stage_counts():
    result = {
        "structured_memory": "API latency is 250ms",
        "compactor_metrics": None,
        "token_usage": {"updater": {"total_tokens": 10, "calls": 1}},
        "summary": {"overall": {
            "structured_elapsed_seconds": 2.0,
            "structured_memory_stats": {"active_entries": 3, "total_active_entry_chars": 100},
        }},
        "results": {"knowledge_update": [
            _item(context="API latency is 250ms", response="250ms", score=1.0,
                  heuristic=True, judge=True)
        ]},
    }
    aggregate = aggregate_runs([result, result])
    comparison = compare_aggregates(aggregate, aggregate)
    assert aggregate["judge_score_mean"] == 1.0
    assert aggregate["tokens"]["updater_tokens"] == 20
    assert aggregate["active_entries_mean"] == 3
    assert aggregate["selected_memory_ids_mean"] == 1
    assert comparison["judge_score_mean_delta"] == 0.0
    assert comparison["tokens"]["updater_tokens"] == 0

from scripts.check_beam_efficiency import evaluate


def _result(score=0.6, entries=40, agent=70000, updater=170000):
    return {
        "summary": {
            "judge_score": score,
            "structured_memory_stats": {"active_entries": entries},
        },
        "token_usage": {
            "agent": {"total_tokens": agent},
            "updater": {"total_tokens": updater},
        },
    }


def test_efficiency_gate_passes_only_when_quality_and_cost_targets_hold():
    assert evaluate(
        _result(),
        min_judge_score=.58,
        max_active_entries=40,
        max_agent_tokens=71000,
        max_updater_tokens=180000,
    ) == []
    assert "judge_score" in evaluate(
        _result(score=.5),
        min_judge_score=.58,
        max_active_entries=40,
        max_agent_tokens=71000,
        max_updater_tokens=180000,
    )


def test_efficiency_gate_accepts_runner_nested_overall_summary():
    result = _result()
    result["summary"] = {"overall": result["summary"]}
    assert evaluate(
        result,
        min_judge_score=.58,
        max_active_entries=40,
        max_agent_tokens=71000,
        max_updater_tokens=180000,
    ) == []


def test_efficiency_gate_checks_memory_content_density():
    result = _result()
    result["summary"]["structured_memory_stats"].update({
        "total_active_entry_chars": 5000,
        "avg_active_entry_chars": 140,
        "long_active_entries_over_180_chars": 8,
    })
    assert evaluate(
        result,
        min_judge_score=.58,
        max_active_entries=40,
        max_agent_tokens=71000,
        max_updater_tokens=180000,
        max_memory_chars=5500,
        max_avg_entry_chars=150,
        max_long_entries=10,
    ) == []

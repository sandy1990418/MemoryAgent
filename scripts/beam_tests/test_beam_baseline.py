import json

from evaluation.beam.regression_report import aggregate_runs
from scripts.run_beam_baseline import build_recent_pair_payload, discover_case_dirs


def _pair(number: int) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"user-{number}"},
        {"role": "assistant", "content": f"assistant-{number}"},
    ]


def test_recent_pair_payload_keeps_latest_pair_last():
    messages = [message for number in range(12) for message in _pair(number)]
    chat = [{"turns": [messages]}]

    payload = build_recent_pair_payload(chat)

    assert len(payload) == 10
    assert payload[0] == {"user": "user-2", "assistant": "assistant-2"}
    assert payload[-1] == {"user": "user-11", "assistant": "assistant-11"}


def test_recent_pair_payload_skips_incomplete_pair():
    chat = [{"turns": [[*_pair(1), {"role": "user", "content": "unanswered"}]]}]

    payload = build_recent_pair_payload(chat)

    assert payload == [{"user": "user-1", "assistant": "assistant-1"}]


def _write_case(root, case_id: int) -> None:
    case_dir = root / str(case_id)
    (case_dir / "probing_questions").mkdir(parents=True)
    (case_dir / "chat.json").write_text("[]", encoding="utf-8")
    (case_dir / "topic.json").write_text("{}", encoding="utf-8")
    (case_dir / "probing_questions" / "probing_questions.json").write_text(
        json.dumps({"abstention": []}), encoding="utf-8"
    )


def test_discover_case_dirs_selects_all_cases_in_numeric_order(tmp_path):
    for case_id in (10, 2, 1):
        _write_case(tmp_path, case_id)

    cases = discover_case_dirs(tmp_path)

    assert [path.name for path in cases] == ["1", "2", "10"]


def test_discover_case_dirs_supports_filters(tmp_path):
    for case_id in range(1, 6):
        _write_case(tmp_path, case_id)

    cases = discover_case_dirs(tmp_path, start_case=2, end_case=4, max_cases=2)

    assert [path.name for path in cases] == ["2", "3"]


def test_aggregate_accepts_standalone_baseline_result():
    result = {
        "baseline_payload": [{"user": "latest fact", "assistant": "noted"}],
        "structured_memory": None,
        "summary": {"overall": {"structured_memory_stats": {}}},
        "token_usage": {
            "agent": {"total_tokens": 12, "calls": 1},
            "judge": {"total_tokens": 8, "calls": 1},
            "updater": {"total_tokens": 0, "calls": 0},
            "compactor": {"total_tokens": 0, "calls": 0},
        },
        "results": {},
    }

    aggregate = aggregate_runs([result])

    assert aggregate["runs"] == 1
    assert aggregate["tokens"]["agent_tokens"] == 12
    assert aggregate["tokens"]["judge_tokens"] == 8

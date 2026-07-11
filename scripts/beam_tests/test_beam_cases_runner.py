import json
import sys

import pytest

from scripts.beam_models import DEFAULT_BEAM_QUESTION_TYPES
from scripts.run_beam_cases import (
    discover_case_dirs,
    parse_args,
    replay_snapshot_lookup,
    resolve_replay_snapshot,
)


def _write_case(root, case_id: int) -> None:
    case_dir = root / str(case_id)
    (case_dir / "probing_questions").mkdir(parents=True)
    (case_dir / "chat.json").write_text("[]", encoding="utf-8")
    (case_dir / "topic.json").write_text(
        json.dumps({"id": case_id}),
        encoding="utf-8",
    )
    (case_dir / "probing_questions" / "probing_questions.json").write_text(
        "{}",
        encoding="utf-8",
    )


def test_discover_case_dirs_sorts_numerically_and_caps(tmp_path):
    for case_id in [10, 2, 1]:
        _write_case(tmp_path, case_id)

    cases = discover_case_dirs(tmp_path, max_cases=2)

    assert [case.name for case in cases] == ["1", "2"]


def test_discover_case_dirs_filters_explicit_case_ids(tmp_path):
    for case_id in [1, 2, 3]:
        _write_case(tmp_path, case_id)

    cases = discover_case_dirs(tmp_path, case_ids=["3", "1"])

    assert [case.name for case in cases] == ["1", "3"]


def test_batch_cli_defaults_to_focused_memory_abilities(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_beam_cases.py"])

    args = parse_args()

    assert tuple(args.question_types) == DEFAULT_BEAM_QUESTION_TYPES


def test_batch_cli_can_select_all_question_types(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_beam_cases.py", "--all-question-types"],
    )

    args = parse_args()

    assert args.question_types is None


def test_batch_cli_reads_yaml_defaults(tmp_path, monkeypatch):
    case_path = tmp_path / "100K" / "7"
    config_path = tmp_path / "beam.yaml"
    config_path.write_text(
        f"data_path: {case_path}\nabilities:\n  - abstention\n"
        "judge: false\nmax_questions_per_type: 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_beam_cases.py", "--beam-config", str(config_path)],
    )

    args = parse_args()

    assert args.case_root == case_path.parent
    assert args.question_types == ["abstention"]
    assert args.max_questions_per_type == 1
    assert args.judge_model is None


def test_batch_cli_accepts_memory_profile(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_beam_cases.py", "--memory-profile", "eval"],
    )

    args = parse_args()

    assert args.memory_profile == "eval"


def test_batch_cli_accepts_frozen_split_repeats_and_baseline(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_beam_cases.py", "--split", "validation", "--repeats", "3",
            "--baseline-manifest", "baseline.json",
        ],
    )

    args = parse_args()

    assert args.split == "validation"
    assert args.repeats == 3
    assert str(args.baseline_manifest) == "baseline.json"
    assert args.routing_mode == "production"


def test_batch_cli_accepts_replay_manifest(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["run_beam_cases.py", "--replay-manifest", "baseline.json"]
    )

    args = parse_args()

    assert str(args.replay_manifest) == "baseline.json"


def _baseline_manifest() -> dict:
    return {
        "cases": [
            {"case_id": "1", "repeat": 1, "status": "ok", "memory_snapshot": "a/1-r1.json"},
            {"case_id": "1", "repeat": 2, "status": "ok", "memory_snapshot": "a/1-r2.json"},
            {"case_id": "4", "repeat": 1, "status": "error"},
            {"case_id": "5", "repeat": 1, "status": "ok", "memory_snapshot": None},
        ]
    }


def test_replay_lookup_maps_case_and_repeat_to_ok_snapshots_only():
    lookup = replay_snapshot_lookup(_baseline_manifest())

    assert lookup == {("1", 1): "a/1-r1.json", ("1", 2): "a/1-r2.json"}


def test_resolve_replay_snapshot_prefers_exact_repeat_then_falls_back():
    lookup = replay_snapshot_lookup(_baseline_manifest())

    assert resolve_replay_snapshot(lookup, "1", 2) == "a/1-r2.json"
    assert resolve_replay_snapshot(lookup, "1", 9) == "a/1-r1.json"
    with pytest.raises(FileNotFoundError, match="case 4"):
        resolve_replay_snapshot(lookup, "4", 1)

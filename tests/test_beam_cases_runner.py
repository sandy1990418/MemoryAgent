import json

from scripts.run_beam_cases import discover_case_dirs


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

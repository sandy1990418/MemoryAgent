from scripts.run_beam_case import rubric_hit


def test_rubric_hit_does_not_use_numeric_shortcut_without_numbers():
    check = rubric_hit(
        response="alpha beta gamma delta",
        rubric_line="LLM response should mention: alpha beta gamma delta epsilon zeta omega",
    )

    assert check["word_overlap_ratio"] < 0.65
    assert check["hit"] is False


def test_rubric_hit_allows_numeric_shortcut_when_number_is_required():
    check = rubric_hit(
        response="Latency was 250ms with SQL caching",
        rubric_line=(
            "LLM response should mention: API latency improved to 250ms "
            "after SQL and caching work"
        ),
    )

    assert check["word_overlap_ratio"] < 0.65
    assert check["hit"] is True

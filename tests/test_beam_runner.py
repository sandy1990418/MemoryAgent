from scripts.run_beam_case import (
    answer_question,
    judge_response,
    normalize_judge_checks,
    parse_judge_response,
    rubric_hit,
)


class FakeJudgeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def complete(self, system, messages, model=None):
        self.calls.append({"system": system, "messages": messages, "model": model})
        return self.response


class FakeAnswerLLM(FakeJudgeLLM):
    pass


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


def test_parse_judge_response_accepts_fenced_json_object():
    parsed = parse_judge_response(
        '```json\n{"checks": [{"passed": true, "reason": "contains date"}]}\n```'
    )

    assert parsed == {"checks": [{"passed": True, "reason": "contains date"}]}


def test_normalize_judge_checks_preserves_rubric_order_and_fills_missing_checks():
    checks = normalize_judge_checks(
        {"checks": [{"passed": True, "reason": "ok"}]},
        [
            "LLM response should mention: Flask 2.3.1",
            "LLM response should mention: SQLite 3.39",
        ],
    )

    assert checks == [
        {
            "rubric": "LLM response should mention: Flask 2.3.1",
            "target": "Flask 2.3.1",
            "passed": True,
            "reason": "ok",
        },
        {
            "rubric": "LLM response should mention: SQLite 3.39",
            "target": "SQLite 3.39",
            "passed": False,
            "reason": "",
        },
    ]


def test_judge_response_returns_normalized_checks_from_llm_json():
    llm = FakeJudgeLLM(
        '{"checks": ['
        '{"passed": true, "reason": "mentions Flask"},'
        '{"passed": false, "reason": "missing SQLite"}'
        "]}"
    )

    checks = judge_response(
        llm=llm,
        model="judge-model",
        question_type="information_extraction",
        question="Which libraries are used?",
        reference="Flask and SQLite",
        response="The project uses Flask.",
        rubric_lines=[
            "LLM response should mention: Flask",
            "LLM response should mention: SQLite",
        ],
    )

    assert [check["passed"] for check in checks] == [True, False]
    assert llm.calls[0]["model"] == "judge-model"
    assert "impartial evaluator" in llm.calls[0]["system"]


def test_answer_question_prompt_requires_supported_concise_answers():
    llm = FakeAnswerLLM("ok")

    response = answer_question(
        llm=llm,
        model="answer-model",
        topic={"title": "Budget tracker"},
        question_type="abstention",
        question="Can you tell me about my previous projects?",
        context="No retrieved memory.",
    )

    assert response == "ok"
    assert llm.calls[0]["model"] == "answer-model"
    system = llm.calls[0]["system"]
    assert "Do not infer background, previous projects, user feedback" in system
    assert "there is contradictory information" in system
    assert "obey any requested item count exactly" in system
    assert "Keep answers concise" in system

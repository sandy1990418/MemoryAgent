import sys
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from memory_agent.models.beam import DEFAULT_BEAM_QUESTION_TYPES, BeamRunConfig
from memory_agent.models.sections import CHAT_SECTIONS
from memory_agent.structured.memory import Memory
from memory_agent.structured.selector import MemorySelector
from scripts.run_beam_case import (
    answer_question,
    beam_answers_from_results,
    beam_evaluation_from_results,
    build_answer_context,
    judge_response,
    load_topic,
    normalize_judge_checks,
    parse_args,
    parse_judge_response,
    rubric_hit,
    select_probes,
)


class FakeJudgeLLM:
    def __init__(self, response: str | list[str]) -> None:
        self.responses = [response] if isinstance(response, str) else list(response)
        self.calls = []

    def complete(self, system, messages, model=None):
        self.calls.append({"system": system, "messages": messages, "model": model})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


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
        {"checks": [{"score": 0.5, "reason": "partial"}]},
        [
            "LLM response should mention: Flask 2.3.1",
            "LLM response should mention: SQLite 3.39",
        ],
    )

    assert checks == [
        {
            "rubric": "LLM response should mention: Flask 2.3.1",
            "target": "Flask 2.3.1",
            "score": 0.5,
            "passed": False,
            "reason": "partial",
        },
        {
            "rubric": "LLM response should mention: SQLite 3.39",
            "target": "SQLite 3.39",
            "score": 0.0,
            "passed": False,
            "reason": "",
        },
    ]


def test_judge_response_returns_normalized_checks_from_llm_json():
    llm = FakeJudgeLLM(
        [
            '{"score": 1.0, "reason": "mentions Flask"}',
            '{"score": 0.0, "reason": "missing SQLite"}',
        ]
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
    assert [check["score"] for check in checks] == [1.0, 0.0]
    assert len(llm.calls) == 2
    assert llm.calls[0]["model"] == "judge-model"
    assert "expert evaluator" in llm.calls[0]["system"]
    assert "specified RUBRIC CRITERION" in llm.calls[0]["messages"][0]["content"]
    assert "SCORING SCALE" in llm.calls[0]["messages"][0]["content"]


def test_beam_compatible_answers_and_evaluation_shapes():
    results = {
        "information_extraction": [
            {
                "question": "Which framework is used?",
                "llm_response": "The project uses Flask 2.3.1.",
                "judge_checks": [
                    {
                        "rubric": "LLM response should mention: Flask 2.3.1",
                        "score": 1.0,
                        "reason": "explicitly mentions Flask 2.3.1",
                    },
                    {
                        "rubric": "LLM response should mention: SQLite 3.39",
                        "score": 0.0,
                        "reason": "missing SQLite",
                    },
                ],
            }
        ],
        "event_ordering": [
            {
                "question": "List the order.",
                "llm_response": "First came setup, then deployment.",
                "judge_checks": [
                    {"rubric": "setup", "score": 1.0, "reason": "present"},
                    {"rubric": "deployment", "score": 1.0, "reason": "present"},
                ],
            }
        ],
    }

    answers = beam_answers_from_results(results)
    evaluation = beam_evaluation_from_results(results)

    assert answers == {
        "information_extraction": [
            {
                "question": "Which framework is used?",
                "llm_response": "The project uses Flask 2.3.1.",
            }
        ],
        "event_ordering": [
            {
                "question": "List the order.",
                "llm_response": "First came setup, then deployment.",
            }
        ],
    }
    assert evaluation["information_extraction"][0]["llm_judge_score"] == 0.5
    assert evaluation["information_extraction"][0]["llm_judge_responses"] == [
        {
            "rubric": "LLM response should mention: Flask 2.3.1",
            "score": 1.0,
            "reason": "explicitly mentions Flask 2.3.1",
        },
        {
            "rubric": "LLM response should mention: SQLite 3.39",
            "score": 0.0,
            "reason": "missing SQLite",
        },
    ]
    assert evaluation["event_ordering"][0]["tau_norm"] == 1.0


def test_load_topic_accepts_case_topic_json_dict():
    topic = {"id": 7, "title": "Case topic"}

    assert load_topic(topic) == topic


def test_select_probes_filters_types_and_caps_questions():
    probes = {
        "information_extraction": [{"question": "a"}, {"question": "b"}],
        "temporal_reasoning": [{"question": "c"}],
    }

    selected = select_probes(
        probes,
        question_types=["information_extraction"],
        max_questions_per_type=1,
    )

    assert selected == {"information_extraction": [{"question": "a"}]}


def test_beam_config_defaults_to_focused_memory_abilities():
    config = BeamRunConfig()

    assert tuple(config.question_types or ()) == DEFAULT_BEAM_QUESTION_TYPES
    assert "information_extraction" not in config.question_types
    assert "temporal_reasoning" not in config.question_types


def test_beam_cli_defaults_to_focused_memory_abilities(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_beam_case.py"])

    args = parse_args()

    assert tuple(args.question_types) == DEFAULT_BEAM_QUESTION_TYPES


def test_beam_cli_all_question_types_disables_filter(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_beam_case.py", "--all-question-types"],
    )

    args = parse_args()

    assert args.question_types is None


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
    assert llm.calls[0]["system"] == "You are an assistant."
    user_prompt = llm.calls[0]["messages"][0]["content"]
    assert "MUST answer questions using ONLY the information provided" in user_prompt
    assert "Do NOT use your internal knowledge" in user_prompt
    assert "ANSWER REQUIREMENTS" in user_prompt
    assert "use the latest active memory entry" in user_prompt
    assert "identify the relevant dated events" in user_prompt
    assert "Abstain only when no relevant memory entry" in user_prompt
    assert "Only output the answer to the question" in user_prompt
    assert "Can you tell me about my previous projects?" in user_prompt


def test_build_answer_context_includes_chronological_order_block():
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "first topic was Flask routing",
                "provenance": [4],
            },
            {
                "op": "ADD",
                "section": "decisions",
                "text": "second topic was deployment",
                "provenance": [8],
            },
        ]
    )
    structured_middleware = SimpleNamespace(
        memory=memory,
        memory_selector=MemorySelector(),
    )

    context = build_answer_context(
        structured_middleware=structured_middleware,
        active_messages=[HumanMessage(content="recent")],
        hits=[],
        max_hit_chars=500,
        max_active_context_chars=500,
        structured_answer_tokens=500,
        query="What came first?",
    )

    assert "# Chronological Order" in context
    assert "Entries ordered by first mention" in context
    chronological_block = context.split("# Chronological Order", 1)[1].split(
        "# Working Conversation Tail", 1
    )[0]
    assert chronological_block.index("[F1] first topic was Flask routing") < chronological_block.index(
        "[D1] second topic was deployment"
    )

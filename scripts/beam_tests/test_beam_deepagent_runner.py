import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from memory_agent.clients.llm import LangChainTokenCallback, TokenLedger
from memory_agent.adapters.langchain.structured_memory import StructuredMemoryMiddleware
from memory_agent.core.sections import AGENT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.update.updater import MemoryUpdater
from scripts.run_beam_case_deepagent import (
    ask_agent,
    build_agent_system_prompt,
    collect_tool_trace,
    final_ai_text,
    parse_args,
)
from tests.fakes import ScriptedLLM


def _search_call(query: str, call_id: str) -> dict:
    return {
        "name": "search_long_term_memory",
        "args": {"query": query},
        "id": call_id,
        "type": "tool_call",
    }


def test_final_ai_text_returns_last_non_tool_call_answer():
    messages = [
        HumanMessage(content="question"),
        AIMessage(content="", tool_calls=[_search_call("first", "call-1")]),
        ToolMessage(content="hit", tool_call_id="call-1"),
        AIMessage(content="the final answer"),
    ]

    assert final_ai_text(messages) == "the final answer"


def test_final_ai_text_skips_tool_call_steps_and_empty_messages():
    messages = [
        AIMessage(content="earlier draft"),
        AIMessage(content="thinking", tool_calls=[_search_call("q", "call-1")]),
        AIMessage(content="   "),
    ]

    assert final_ai_text(messages) == "earlier draft"


def test_final_ai_text_returns_empty_string_without_answers():
    assert final_ai_text([]) == ""
    assert final_ai_text([HumanMessage(content="question")]) == ""


def test_collect_tool_trace_keeps_only_search_calls_in_order():
    other_call = {"name": "write_todos", "args": {"todos": []}, "id": "call-x", "type": "tool_call"}
    messages = [
        AIMessage(content="", tool_calls=[_search_call("alpha", "call-1"), other_call]),
        ToolMessage(content="hit", tool_call_id="call-1"),
        AIMessage(content="", tool_calls=[_search_call("beta", "call-2")]),
        AIMessage(content="done"),
    ]

    trace = collect_tool_trace(messages)

    assert [entry["args"]["query"] for entry in trace] == ["alpha", "beta"]
    assert all(entry["name"] == "search_long_term_memory" for entry in trace)


def test_deepagent_system_prompt_requires_supported_concise_answers():
    prompt = build_agent_system_prompt(
        structured_middleware=None,
        active_messages=[HumanMessage(content="recent")],
        structured_answer_tokens=500,
        max_active_context_chars=500,
    )

    assert "Use the search_long_term_memory tool when the prompt memory" in prompt
    assert "use a few targeted searches" in prompt
    assert "Use only available memory and, when a tool is available" in prompt
    assert "Be concise" in prompt
    assert "evidence is insufficient" in prompt
    assert "evidence conflicts" in prompt
    assert "Use chronological evidence" in prompt


class FakeAgent:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, state, config=None):
        self.calls.append({"state": state, "config": config})
        return {"messages": [AIMessage(content="ok")]}


def test_ask_agent_includes_chronological_order_with_structured_memory():
    memory = Memory(sections=AGENT_SECTIONS)
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "first topic was Flask routing",
                "provenance": [11],
            },
            {
                "op": "ADD",
                "section": "progress",
                "text": "second topic was deployment",
                "provenance": [22],
            },
        ]
    )
    structured_middleware = StructuredMemoryMiddleware(
        memory=memory,
        updater=MemoryUpdater(
            llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
            sections=AGENT_SECTIONS,
        ),
        max_tokens=1000,
    )
    agent = FakeAgent()

    response, _tool_trace, _message_count = ask_agent(
        agent=agent,
        topic={"title": "Budget tracker"},
        question_type="event_ordering",
        question="Which topic came first?",
        recursion_limit=10,
        structured_middleware=structured_middleware,
        structured_answer_tokens=500,
    )

    assert response == "ok"
    user_message = agent.calls[0]["state"]["messages"][0]["content"]
    assert "# Chronological Order" in user_message
    assert "Entries ordered by first mention" in user_message
    chronological_block = user_message.split("# Chronological Order", 1)[1].split(
        "Probing question:", 1
    )[0]
    assert chronological_block.index("[F1] first topic was Flask routing") < chronological_block.index(
        "[P1] second topic was deployment"
    )

def test_ask_agent_surfaces_recorded_denials_block():
    memory = Memory(sections=AGENT_SECTIONS)
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "status_changes",
                "text": "User stated: I've never integrated Flask-Login into this project.",
                "provenance": [108],
            },
            {
                "op": "ADD",
                "section": "facts",
                "text": "User is integrating Flask-Login v0.6.2 for session management.",
                "provenance": [66],
            },
        ]
    )
    structured_middleware = StructuredMemoryMiddleware(
        memory=memory,
        updater=MemoryUpdater(
            llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
            sections=AGENT_SECTIONS,
        ),
        max_tokens=1000,
    )
    agent = FakeAgent()

    ask_agent(
        agent=agent,
        topic={"title": "Budget tracker"},
        question_type="contradiction_resolution",
        question="Have I integrated Flask-Login?",
        recursion_limit=10,
        structured_middleware=structured_middleware,
        structured_answer_tokens=500,
    )

    user_message = agent.calls[0]["state"]["messages"][0]["content"]
    assert "# Recorded Denials and Corrections" in user_message
    denials_block = user_message.split("# Recorded Denials and Corrections", 1)[1].split(
        "Probing question:", 1
    )[0]
    assert "I've never integrated Flask-Login" in denials_block
    assert "Flask-Login v0.6.2 for session management" not in denials_block
    assert "denying, correcting, or reversing" in denials_block

def test_system_prompt_without_retrieval_omits_search_tool():
    prompt = build_agent_system_prompt(
        structured_middleware=None,
        active_messages=[HumanMessage(content="recent")],
        structured_answer_tokens=500,
        max_active_context_chars=500,
        retrieval_enabled=False,
    )

    assert "search_long_term_memory" not in prompt
    assert "No retrieval tool is available" in prompt
    assert "Use only available memory and, when a tool is available" in prompt
    assert "Be concise" in prompt
    assert "evidence conflicts" in prompt
    assert "Use chronological evidence" in prompt


def test_ask_agent_without_retrieval_instructs_memory_only_answering():
    memory = Memory(sections=AGENT_SECTIONS)
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "project uses Flask",
                "provenance": [3],
            }
        ]
    )
    structured_middleware = StructuredMemoryMiddleware(
        memory=memory,
        updater=MemoryUpdater(
            llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
            sections=AGENT_SECTIONS,
        ),
        max_tokens=1000,
    )
    agent = FakeAgent()

    ask_agent(
        agent=agent,
        topic={"title": "Budget tracker"},
        question_type="information_extraction",
        question="What framework is used?",
        recursion_limit=10,
        structured_middleware=structured_middleware,
        structured_answer_tokens=500,
        retrieval_enabled=False,
    )

    user_message = agent.calls[0]["state"]["messages"][0]["content"]
    assert "Use the search_long_term_memory tool" not in user_message
    assert "Answer using only the memory sections above" in user_message
    assert "# Question-Relevant Structured Memory" in user_message
    assert "# Recorded Denials and Corrections" in user_message


def test_ask_agent_records_agent_token_role():
    agent = FakeAgent()
    ledger = TokenLedger()

    response, _trace, _message_count = ask_agent(
        agent=agent,
        topic={"title": "Budget tracker"},
        question_type="abstention",
        question="What do you know?",
        recursion_limit=10,
        retrieval_enabled=False,
        token_ledger=ledger,
    )

    assert response == "ok"
    assert ledger.to_dict()["agent"]["calls"] == 1


def test_agent_token_callback_records_each_provider_call():
    ledger = TokenLedger()
    callback = LangChainTokenCallback(ledger, "agent")

    callback.on_llm_end(
        SimpleNamespace(
            llm_output={
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 4}
            },
            generations=[],
        )
    )
    callback.on_llm_end(
        SimpleNamespace(
            llm_output={
                "token_usage": {"input_tokens": 6, "output_tokens": 3}
            },
            generations=[],
        )
    )

    assert ledger.to_dict()["agent"] == {
        "input_tokens": 16,
        "output_tokens": 7,
        "total_tokens": 23,
        "calls": 2,
    }


def test_deepagent_cli_reads_yaml_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "beam.yaml"
    config_path.write_text(
        f"data_path: {tmp_path / 'case'}\nabilities:\n  - knowledge_update\n"
        "judge: false\nmax_questions_per_type: 3\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_beam_case_deepagent.py", "--beam-config", str(config_path)],
    )

    args = parse_args()

    assert args.probes == tmp_path / "case" / "probing_questions" / "probing_questions.json"
    assert args.question_types == ["knowledge_update"]
    assert args.max_questions_per_type == 3
    assert args.judge_model is None

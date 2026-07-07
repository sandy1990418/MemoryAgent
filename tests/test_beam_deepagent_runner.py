from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from scripts.run_beam_case_deepagent import collect_tool_trace, final_ai_text


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

import asyncio

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from memory_agent.longterm import LongTermHit, LongTermMemoryMiddleware
from tests.fakes import FakeLongTermMemory


class DummyModel:
    """Stand-in for a BaseChatModel; ModelRequest does no runtime type check."""


def h(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def a(content: str) -> AIMessage:
    return AIMessage(content=content)


def summary(content: str = "summary") -> HumanMessage:
    return HumanMessage(content=content, additional_kwargs={"lc_source": "summarization"})


def request(messages, system_prompt: str = "Base prompt") -> ModelRequest:
    return ModelRequest(model=DummyModel(), messages=messages, system_prompt=system_prompt)


def capture_handler(captured):
    def handler(req):
        captured["request"] = req
        return req

    return handler


def test_first_before_model_call_tracks_but_never_adds():
    long_term = FakeLongTermMemory()
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")

    result = middleware.before_model({"messages": [h("hello"), a("hi")]}, None)

    assert result is None
    assert long_term.add_calls == []


def test_eviction_diff_pushes_disappeared_messages_in_order():
    long_term = FakeLongTermMemory()
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")
    h1, a1, h2, a2, h3, a3 = (
        h("h1"),
        a("a1"),
        h("h2"),
        a("a2"),
        h("h3"),
        a("a3"),
    )

    middleware.before_model({"messages": [h1, a1, h2, a2, h3, a3]}, None)
    middleware.before_model({"messages": [summary(), h3, a3]}, None)

    assert long_term.add_calls == [
        {
            "messages": [
                {"role": "user", "content": "h1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "h2"},
                {"role": "assistant", "content": "a2"},
            ],
            "user_id": "user-1",
            "metadata": {"source": "eviction"},
        }
    ]


def test_summary_message_is_never_pushed_on_eviction_or_flush():
    long_term = FakeLongTermMemory()
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")
    h1, s1, a1 = h("h1"), summary("summarized prior context"), a("a1")

    middleware.before_model({"messages": [h1, s1, a1]}, None)
    middleware.before_model({"messages": [h1, a1]}, None)

    assert long_term.add_calls == []
    assert middleware.flush() == 2
    assert long_term.add_calls == [
        {
            "messages": [
                {"role": "user", "content": "h1"},
                {"role": "assistant", "content": "a1"},
            ],
            "user_id": "user-1",
            "metadata": {"source": "flush"},
        }
    ]


def test_tool_messages_and_tool_call_only_ai_messages_are_never_pushed():
    long_term = FakeLongTermMemory()
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")
    tool_call_only = AIMessage(
        content="",
        tool_calls=[{"name": "weather", "args": {"city": "Taipei"}, "id": "call1"}],
    )
    tool_message = ToolMessage(content="sunny", tool_call_id="call1", name="weather")

    middleware.before_model({"messages": [h("track me"), tool_call_only, tool_message]}, None)

    assert middleware.flush() == 1
    assert long_term.add_calls[0]["messages"] == [{"role": "user", "content": "track me"}]


def test_add_failure_retries_same_eviction_batch_and_never_repushes_after_success():
    long_term = FakeLongTermMemory()
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")
    h1, a1 = h("h1"), a("a1")
    long_term.raise_on_add = RuntimeError("backend down")

    middleware.before_model({"messages": [h1, a1]}, None)
    middleware.before_model({"messages": []}, None)

    assert long_term.add_calls == []

    long_term.raise_on_add = None
    middleware.before_model({"messages": []}, None)
    middleware.before_model({"messages": []}, None)

    assert long_term.add_calls == [
        {
            "messages": [
                {"role": "user", "content": "h1"},
                {"role": "assistant", "content": "a1"},
            ],
            "user_id": "user-1",
            "metadata": {"source": "eviction"},
        }
    ]
    assert middleware.flush() == 0


def test_flush_pushes_remaining_messages_once_and_skips_eviction_pushed_messages():
    long_term = FakeLongTermMemory()
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")
    h1, a1, h2 = h("h1"), a("a1"), h("h2")

    middleware.before_model({"messages": [h1, a1, h2]}, None)
    middleware.before_model({"messages": [h2]}, None)

    assert long_term.add_calls[0]["metadata"] == {"source": "eviction"}
    assert long_term.add_calls[0]["messages"] == [
        {"role": "user", "content": "h1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert middleware.flush() == 1
    assert long_term.add_calls[1] == {
        "messages": [{"role": "user", "content": "h2"}],
        "user_id": "user-1",
        "metadata": {"source": "flush"},
    }
    assert middleware.flush() == 0


def test_wrap_model_call_injects_hits_into_system_prompt_and_calls_handler():
    long_term = FakeLongTermMemory(hits=[LongTermHit("remembered fact", score=0.9)])
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")
    captured = {}

    result = middleware.wrap_model_call(
        request([h("what do you remember?")]),
        capture_handler(captured),
    )

    assert result is captured["request"]
    assert "# Long-Term Memory" in captured["request"].system_prompt
    assert "remembered fact" in captured["request"].system_prompt
    assert long_term.search_calls == [
        {"query": "what do you remember?", "user_id": "user-1", "limit": 5}
    ]
    assert middleware.last_recalled == [LongTermHit("remembered fact", score=0.9)]


def test_wrap_model_call_leaves_prompt_unchanged_when_no_hits():
    long_term = FakeLongTermMemory()
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")
    captured = {}

    middleware.wrap_model_call(request([h("query")]), capture_handler(captured))

    assert captured["request"].system_prompt == "Base prompt"
    assert middleware.last_recalled == []


def test_wrap_model_call_search_failure_does_not_raise_or_modify_prompt():
    long_term = FakeLongTermMemory()
    long_term.raise_on_search = RuntimeError("backend down")
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")
    captured = {}

    middleware.wrap_model_call(request([h("query")]), capture_handler(captured))

    assert captured["request"].system_prompt == "Base prompt"
    assert middleware.last_recalled == []


def test_query_selection_uses_latest_non_summary_human_message():
    long_term = FakeLongTermMemory(hits=[LongTermHit("stored fact")])
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")

    middleware.wrap_model_call(
        request(
            [
                h("old query"),
                summary("older summary"),
                a("assistant reply"),
                h("real query"),
                summary("newer summary"),
            ]
        ),
        lambda req: req,
    )

    assert long_term.search_calls == [
        {"query": "real query", "user_id": "user-1", "limit": 5}
    ]


def test_search_cache_reuses_same_query_and_searches_different_query():
    long_term = FakeLongTermMemory(hits=[LongTermHit("stored fact")])
    middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")

    middleware.wrap_model_call(request([h("same query")]), lambda req: req)
    middleware.wrap_model_call(request([h("same query")]), lambda req: req)
    middleware.wrap_model_call(request([h("different query")]), lambda req: req)

    assert long_term.search_calls == [
        {"query": "same query", "user_id": "user-1", "limit": 5},
        {"query": "different query", "user_id": "user-1", "limit": 5},
    ]


def test_max_memory_tokens_injects_fewer_hits_than_available():
    long_term = FakeLongTermMemory(
        hits=[
            LongTermHit("tiny"),
            LongTermHit("this memory is too long for the tiny budget"),
        ]
    )
    middleware = LongTermMemoryMiddleware(
        long_term,
        user_id="user-1",
        max_memory_tokens=2,
    )
    captured = {}

    middleware.wrap_model_call(request([h("query")]), capture_handler(captured))

    prompt = captured["request"].system_prompt
    assert "tiny" in prompt
    assert "this memory is too long" not in prompt


def test_async_before_model_and_wrap_model_call():
    async def run():
        long_term = FakeLongTermMemory(hits=[LongTermHit("async memory")])
        middleware = LongTermMemoryMiddleware(long_term, user_id="user-1")
        captured = {}

        result = await middleware.abefore_model({"messages": [h("hello")]}, None)

        async def handler(req):
            captured["request"] = req
            return "ok"

        wrapped = await middleware.awrap_model_call(request([h("async query")]), handler)
        assert result is None
        assert wrapped == "ok"
        assert "async memory" in captured["request"].system_prompt

    asyncio.run(run())

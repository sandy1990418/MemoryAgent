from memory_agent.llm import OpenAIClient


class FakeChatModel:
    """Minimal `BaseChatModel`-like fake: records `invoke` calls, returns a
    canned response. Does not implement `.text` so tests exercise
    `OpenAIClient._extract_text`'s manual `.content` fallback path.
    """

    def __init__(self, content) -> None:
        self.content = content
        self.invoke_calls: list[list[dict]] = []

    def invoke(self, messages: list[dict]):
        self.invoke_calls.append(messages)
        return FakeResponse(self.content)


class FakeResponse:
    """Response double with only `.content`, no `.text` property."""

    def __init__(self, content) -> None:
        self.content = content


def make_client(model: str, chat_models: dict[str, FakeChatModel]) -> OpenAIClient:
    return OpenAIClient(model, chat_model_factory=lambda resolved: chat_models[resolved])


def test_openai_prefix_is_stripped_before_resolving_chat_model():
    fake = FakeChatModel("hello")
    client = make_client("openai:gpt-5.4-nano", {"gpt-5.4-nano": fake})

    result = client.complete("system prompt", [{"role": "user", "content": "hi"}])

    assert result == "hello"
    assert len(fake.invoke_calls) == 1


def test_system_and_messages_are_assembled_into_full_message_list():
    fake = FakeChatModel("ok")
    client = make_client("gpt-5.4-nano", {"gpt-5.4-nano": fake})
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    client.complete("be helpful", messages)

    assert fake.invoke_calls == [
        [{"role": "system", "content": "be helpful"}] + messages
    ]


def test_text_extraction_handles_plain_string_content():
    fake = FakeChatModel("plain string response")
    client = make_client("gpt-5.4-nano", {"gpt-5.4-nano": fake})

    result = client.complete("sys", [])

    assert result == "plain string response"


def test_text_extraction_handles_list_of_text_blocks():
    fake = FakeChatModel(
        [
            {"type": "text", "text": "part one "},
            {"type": "text", "text": "part two"},
            {"type": "reasoning", "text": "should be ignored"},
        ]
    )
    client = make_client("gpt-5.4-nano", {"gpt-5.4-nano": fake})

    result = client.complete("sys", [])

    assert result == "part one part two"


def test_text_extraction_returns_empty_string_when_content_is_empty():
    fake = FakeChatModel([])
    client = make_client("gpt-5.4-nano", {"gpt-5.4-nano": fake})

    assert client.complete("sys", []) == ""


def test_per_call_model_override_selects_a_different_cached_chat_model():
    default_fake = FakeChatModel("from default model")
    override_fake = FakeChatModel("from override model")
    client = make_client(
        "openai:gpt-5.4-nano",
        {"gpt-5.4-nano": default_fake, "gpt-5.4": override_fake},
    )

    default_result = client.complete("sys", [])
    override_result = client.complete("sys", [], model="openai:gpt-5.4")

    assert default_result == "from default model"
    assert override_result == "from override model"
    assert len(default_fake.invoke_calls) == 1
    assert len(override_fake.invoke_calls) == 1


def test_chat_models_are_cached_per_resolved_model_name():
    calls: list[str] = []

    def factory(resolved: str) -> FakeChatModel:
        calls.append(resolved)
        return FakeChatModel("cached")

    client = OpenAIClient("gpt-5.4-nano", chat_model_factory=factory)

    client.complete("sys", [])
    client.complete("sys", [])

    assert calls == ["gpt-5.4-nano"]


def test_default_constructor_does_not_require_langchain_openai_to_be_imported():
    # Constructing the client (without calling complete()) must not import
    # langchain_openai eagerly -- that only happens lazily inside
    # OpenAIClient._build_chat_model on first use with the real factory.
    client = OpenAIClient("gpt-5.4-nano")

    assert client.model == "gpt-5.4-nano"
    assert client._chat_models == {}

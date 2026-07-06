from memory_agent.longterm import Mem0LongTermMemory, build_local_config


class FakeMem0Client:
    def __init__(self, search_result=None) -> None:
        self.search_result = search_result if search_result is not None else {"results": []}
        self.add_calls = []
        self.search_calls = []

    def add(self, messages, **kwargs):
        self.add_calls.append({"messages": messages, "kwargs": kwargs})

    def search(self, query, **kwargs):
        self.search_calls.append({"query": query, "kwargs": kwargs})
        return self.search_result


def test_oss_add_passes_keyword_only_user_id_metadata_and_infer():
    client = FakeMem0Client()
    memory = Mem0LongTermMemory(client, infer=False)
    messages = [{"role": "user", "content": "remember this"}]
    metadata = {"source": "test"}

    memory.add(messages, user_id="user-1", metadata=metadata)

    assert client.add_calls == [
        {
            "messages": messages,
            "kwargs": {"user_id": "user-1", "metadata": metadata, "infer": False},
        }
    ]


def test_platform_add_uses_filters_without_infer_or_user_id_kwargs():
    client = FakeMem0Client()
    memory = Mem0LongTermMemory(client, platform=True, infer=False)
    messages = [{"role": "assistant", "content": "stored fact"}]
    metadata = {"source": "platform-test"}

    memory.add(messages, user_id="user-1", metadata=metadata)

    assert client.add_calls == [
        {
            "messages": messages,
            "kwargs": {"filters": {"user_id": "user-1"}, "metadata": metadata},
        }
    ]


def test_search_uses_top_k_filters_and_normalizes_dict_results():
    client = FakeMem0Client(
        {
            "results": [
                {"memory": "first memory", "score": 0.9, "metadata": {"kind": "fact"}},
                {"memory": "", "score": 0.8, "metadata": {"skip": True}},
                {"score": 0.7, "metadata": {"skip": True}},
                {"memory": None, "score": 0.6, "metadata": {"skip": True}},
                {"memory": 123, "score": 0.5, "metadata": {"skip": True}},
                {"memory": "   ", "score": 0.4, "metadata": {"skip": True}},
                {"memory": "second memory", "score": None, "metadata": None},
            ]
        }
    )
    memory = Mem0LongTermMemory(client)

    hits = memory.search("query", user_id="user-1", limit=3)

    assert client.search_calls == [
        {"query": "query", "kwargs": {"top_k": 3, "filters": {"user_id": "user-1"}}}
    ]
    assert [hit.text for hit in hits] == ["first memory", "second memory"]
    assert hits[0].score == 0.9
    assert hits[0].metadata == {"kind": "fact"}
    assert hits[1].score is None
    assert hits[1].metadata is None


def test_search_tolerates_bare_list_and_empty_results():
    client = FakeMem0Client(
        [
            {"memory": "bare memory", "score": 0.3, "metadata": {"source": "list"}},
            {"memory": ""},
        ]
    )
    memory = Mem0LongTermMemory(client)

    hits = memory.search("query", user_id="user-1")

    assert [hit.text for hit in hits] == ["bare memory"]
    assert hits[0].score == 0.3
    assert hits[0].metadata == {"source": "list"}

    client.search_result = {"results": []}
    assert memory.search("other query", user_id="user-1") == []


def test_build_local_config_returns_expected_dict_without_llm_block():
    config = build_local_config(
        data_dir="data",
        collection_name="collection",
        embedder_model="text-embedding-3-small",
        embedding_dims=1536,
    )

    assert config == {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "collection",
                "path": "data/qdrant",
                "on_disk": True,
                "embedding_model_dims": 1536,
            },
        },
        "history_db_path": "data/history.db",
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
    }
    assert "llm" not in config


def test_build_local_config_includes_llm_block_only_when_model_is_set():
    config = build_local_config(
        data_dir="data",
        collection_name="collection",
        llm_model="gpt-test",
        embedder_model="embed-test",
        embedding_dims=768,
    )

    assert config == {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "collection",
                "path": "data/qdrant",
                "on_disk": True,
                "embedding_model_dims": 768,
            },
        },
        "history_db_path": "data/history.db",
        "embedder": {
            "provider": "openai",
            "config": {"model": "embed-test"},
        },
        "llm": {
            "provider": "openai",
            "config": {"model": "gpt-test"},
        },
    }

from collections.abc import Callable


class ScriptedLLM:
    """Deterministic fake LLM for tests: delegates to a callback."""

    def __init__(self, script: Callable[[str, list[dict]], str]) -> None:
        self.script = script

    def complete(self, system: str, messages: list[dict], model: str | None = None) -> str:
        return self.script(system, messages)


class FakeLongTermMemory:
    """In-memory fake of the LongTermMemory protocol.

    Records add/search calls and returns canned hits. Set `raise_on_add` /
    `raise_on_search` to simulate a failing backend.
    """

    def __init__(self, hits: list | None = None) -> None:
        self.hits = list(hits or [])
        self.add_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self.raise_on_add: Exception | None = None
        self.raise_on_search: Exception | None = None

    def add(self, messages: list[dict], user_id: str, metadata: dict | None = None) -> None:
        if self.raise_on_add is not None:
            raise self.raise_on_add
        self.add_calls.append({"messages": messages, "user_id": user_id, "metadata": metadata})

    def search(self, query: str, user_id: str, limit: int = 5) -> list:
        if self.raise_on_search is not None:
            raise self.raise_on_search
        self.search_calls.append({"query": query, "user_id": user_id, "limit": limit})
        return list(self.hits[:limit])

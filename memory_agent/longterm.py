"""Long-term vector memory abstraction backed by mem0.

The core package stays free of network dependencies; `mem0` is only imported
lazily inside the factory classmethods of the adapter, mirroring how
`OpenAIClient` treats `openai` in `memory_agent/llm.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LongTermHit:
    """One memory recalled from the long-term vector store."""

    text: str
    score: float | None = None
    metadata: dict | None = None


class LongTermMemory(Protocol):
    """Minimal interface the rest of the package depends on.

    `messages` is a list of `{"role": ..., "content": ...}` dicts (roles
    "user" / "assistant"). Implementations decide how to distill and store
    them. `search` returns the most relevant stored memories for a query.
    """

    def add(self, messages: list[dict], user_id: str, metadata: dict | None = None) -> None:
        ...

    def search(self, query: str, user_id: str, limit: int = 5) -> list[LongTermHit]:
        ...


def build_local_config(
    data_dir: str = ".mem0",
    collection_name: str = "mem0",
    llm_model: str | None = None,
    embedder_model: str = "text-embedding-3-small",
    embedding_dims: int = 1536,
) -> dict:
    """Build a local persistent mem0 OSS config without importing mem0."""
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "path": os.path.join(data_dir, "qdrant"),
                "on_disk": True,
                "embedding_model_dims": embedding_dims,
            },
        },
        "history_db_path": os.path.join(data_dir, "history.db"),
        "embedder": {
            "provider": "openai",
            "config": {"model": embedder_model},
        },
    }
    if llm_model is not None:
        config["llm"] = {
            "provider": "openai",
            "config": {"model": llm_model},
        }
    return config


class Mem0LongTermMemory:
    """Adapter from mem0's client API to the local `LongTermMemory` protocol."""

    def __init__(self, client: Any, platform: bool = False, infer: bool = True) -> None:
        self._client = client
        self._platform = platform
        self._infer = infer

    @classmethod
    def from_local(
        cls,
        data_dir: str = ".mem0",
        collection_name: str = "mem0",
        llm_model: str | None = None,
        embedder_model: str = "text-embedding-3-small",
        embedding_dims: int = 1536,
        infer: bool = True,
    ) -> "Mem0LongTermMemory":
        """Create a local persistent OSS mem0 client."""
        from mem0 import Memory

        config = build_local_config(
            data_dir=data_dir,
            collection_name=collection_name,
            llm_model=llm_model,
            embedder_model=embedder_model,
            embedding_dims=embedding_dims,
        )
        return cls(Memory.from_config(config), platform=False, infer=infer)

    @classmethod
    def from_platform(
        cls,
        api_key: str | None = None,
        infer: bool = True,
    ) -> "Mem0LongTermMemory":
        """Create a hosted mem0 platform client."""
        from mem0 import MemoryClient

        client = MemoryClient(api_key=api_key) if api_key is not None else MemoryClient()
        return cls(client, platform=True, infer=infer)

    def add(self, messages: list[dict], user_id: str, metadata: dict | None = None) -> None:
        if self._platform:
            self._client.add(messages, filters={"user_id": user_id}, metadata=metadata)
            return
        self._client.add(messages, user_id=user_id, metadata=metadata, infer=self._infer)

    def search(self, query: str, user_id: str, limit: int = 5) -> list[LongTermHit]:
        raw = self._client.search(query, top_k=limit, filters={"user_id": user_id})
        if isinstance(raw, dict):
            results = raw.get("results", [])
        elif isinstance(raw, list):
            results = raw
        else:
            results = []

        hits: list[LongTermHit] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            text = item.get("memory")
            if not isinstance(text, str) or not text.strip():
                continue
            metadata = item.get("metadata")
            hits.append(
                LongTermHit(
                    text=text,
                    score=item.get("score"),
                    metadata=metadata if isinstance(metadata, dict) else None,
                )
            )
        return hits

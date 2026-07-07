"""LangChain middleware for mem0-style long-term vector recall.

This module is opt-in: it is never imported from `memory_agent/__init__.py`, so
the core package stays framework-free. It is designed to sit after LangChain's
`SummarizationMiddleware` in the middleware list. LangChain applies earlier
`before_model` state updates before later middleware runs in the same model-call
cycle, so this middleware can see the compacted message list immediately and
persist the message IDs that disappeared from the active context window.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from memory_agent.clients.mem0 import LongTermMemory
from memory_agent.models.longterm import LongTermHit

logger = logging.getLogger(__name__)


def _char_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                text = block.get("text")
                parts.append(text if isinstance(text, str) else str(block))
                continue
            text_attr = getattr(block, "text", None)
            parts.append(text_attr if isinstance(text_attr, str) else str(block))
        return "\n".join(parts)
    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(content)


def _is_summary_message(message: AnyMessage) -> bool:
    return getattr(message, "additional_kwargs", {}).get("lc_source") == "summarization"


def _ensure_ids(messages: list[AnyMessage]) -> None:
    for message in messages:
        if message.id is None:
            message.id = str(uuid.uuid4())


class LongTermMemoryMiddleware(AgentMiddleware):
    """Persists summarized-away messages and injects semantic long-term recall."""

    def __init__(
        self,
        long_term: LongTermMemory,
        user_id: str,
        search_limit: int = 5,
        max_memory_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self._long_term = long_term
        self._user_id = user_id
        self._search_limit = search_limit
        self._max_memory_tokens = max_memory_tokens
        self._tracked: dict[str, dict] = {}
        self._summary_ids: set[str] = set()
        self._pushed_ids: set[str] = set()
        self._search_cache: tuple[str, tuple[LongTermHit, ...]] | None = None
        self.last_recalled: list[LongTermHit] = []

    @staticmethod
    def _message_payload(message: AnyMessage) -> dict | None:
        if _is_summary_message(message):
            return None
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            return None

        content = _content_to_text(message.content).strip()
        if not content:
            return None
        return {"role": role, "content": content}

    def _track_current_messages(self, messages: list[AnyMessage]) -> None:
        for message in messages:
            if message.id is None:
                continue
            if _is_summary_message(message):
                self._summary_ids.add(message.id)
                continue
            if message.id in self._tracked:
                continue
            payload = self._message_payload(message)
            if payload is not None:
                self._tracked[message.id] = payload

    def _unpushed_batch(self, ids: list[str]) -> list[dict]:
        return [
            self._tracked[message_id]
            for message_id in ids
            if (
                message_id in self._tracked
                and message_id not in self._summary_ids
                and message_id not in self._pushed_ids
            )
        ]

    def _push_batch(self, ids: list[str], metadata: dict) -> int:
        batch = self._unpushed_batch(ids)
        if not batch:
            return 0

        try:
            self._long_term.add(batch, self._user_id, metadata=metadata)
        except Exception as exc:
            logger.warning("Long-term memory add failed; will retry later: %s", exc)
            return 0

        for message_id in ids:
            if message_id in self._tracked and message_id not in self._summary_ids:
                self._pushed_ids.add(message_id)
        return len(batch)

    def before_model(self, state: AgentState, runtime: Any) -> None:
        messages: list[AnyMessage] = state["messages"]
        _ensure_ids(messages)

        current_ids = {message.id for message in messages if message.id is not None}
        disappeared = [
            message_id
            for message_id in self._tracked
            if message_id not in current_ids and message_id not in self._pushed_ids
        ]
        self._push_batch(disappeared, metadata={"source": "eviction"})
        self._track_current_messages(messages)
        return None

    async def abefore_model(self, state: AgentState, runtime: Any) -> None:
        return self.before_model(state, runtime)

    @staticmethod
    def _query_from_messages(messages: list[AnyMessage]) -> str:
        for message in reversed(messages):
            if _is_summary_message(message):
                continue
            if isinstance(message, HumanMessage):
                text = _content_to_text(message.content).strip()
                if text:
                    return text
        return ""

    def _search(self, query: str) -> list[LongTermHit]:
        if self._search_cache is not None and self._search_cache[0] == query:
            return list(self._search_cache[1])

        try:
            hits = self._long_term.search(query, self._user_id, limit=self._search_limit)
        except Exception as exc:
            logger.warning("Long-term memory search failed; skipping recall: %s", exc)
            return []

        self._search_cache = (query, tuple(hits))
        return list(hits)

    def _memory_lines(self, hits: list[LongTermHit]) -> list[str]:
        lines: list[str] = []
        used_tokens = 0
        for hit in hits:
            line = f"- {hit.text}"
            if self._max_memory_tokens is not None:
                next_tokens = _char_token_estimator(line)
                if used_tokens + next_tokens > self._max_memory_tokens:
                    break
                used_tokens += next_tokens
            lines.append(line)
        return lines

    @staticmethod
    def _with_system_prompt(request: "ModelRequest[Any]", system_prompt: str) -> "ModelRequest[Any]":
        if hasattr(request, "override"):
            return request.override(system_prompt=system_prompt)
        request.system_prompt = system_prompt
        return request

    def _prepare_request(self, request: "ModelRequest[Any]") -> "ModelRequest[Any]":
        query = self._query_from_messages(list(request.messages))
        if not query:
            self.last_recalled = []
            return request

        hits = self._search(query)
        self.last_recalled = hits
        if not hits:
            return request

        lines = self._memory_lines(hits)
        if not lines:
            return request

        block = (
            "# Long-Term Memory\n"
            "Relevant memories from earlier sessions:\n"
            + "\n".join(lines)
        )
        new_prompt = (request.system_prompt or "") + "\n\n" + block
        return self._with_system_prompt(request, new_prompt)

    def wrap_model_call(self, request: "ModelRequest[Any]", handler: Callable[..., Any]) -> Any:
        request = self._prepare_request(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: "ModelRequest[Any]",
        handler: Callable[..., Any],
    ) -> Any:
        request = self._prepare_request(request)
        return await handler(request)

    def flush(self) -> int:
        ids = [
            message_id
            for message_id in self._tracked
            if message_id not in self._summary_ids and message_id not in self._pushed_ids
        ]
        return self._push_batch(ids, metadata={"source": "flush"})

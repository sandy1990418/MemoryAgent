"""Optional LangChain chat adapter for structured conversation memory.

The production memory implementation is framework neutral.  This module is
the deliberately small integration layer for applications that use
``langchain-core`` chat messages.  It is *not* an agent middleware: there are
no imports from ``langchain.agents`` or LangGraph and no tool execution
semantics here.

The adapter owns a local list of chat messages.  Before each model call it
evicts a safe prefix into structured memory and injects selected memory into a
``SystemMessage``.  A prefix is removed only after the structured update has
committed.  Failed/rejected updates therefore leave the exact same messages in
the local history and are retried on the next call.

Tool messages are retained in the local history as part of the model protocol,
but are intentionally not mirrored into durable memory.  AI tool-call
messages and their consecutive ``ToolMessage`` responses are treated as one
eviction unit so a model never receives a dangling tool response.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Sequence
from typing import Any, Callable

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately

from memory_agent.application.structured_service import StructuredMemoryService
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.core.transcript_store import Transcript
from memory_agent.policies.structured import StructuredMemoryPolicy
from memory_agent.retrieval.context import build_answer_memory_context
from memory_agent.retrieval.selector import MemorySelector
from memory_agent.update.compactor import MemoryCompactor
from memory_agent.update.updater import MemoryUpdater
from memory_agent.update.verifier import MemoryUpdateVerifier

logger = logging.getLogger(__name__)

TokenCounter = Callable[[list[AnyMessage]], int]


def _char_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


def _content_to_text(content: Any) -> str:
    """Extract readable text from the content shapes supported by LangChain."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                parts.append(text if isinstance(text, str) else str(block))
            else:
                text = getattr(block, "text", None)
                parts.append(text if isinstance(text, str) else str(block))
        return "\n".join(parts)
    text = getattr(content, "text", None)
    return text if isinstance(text, str) else str(content)


def _message_to_turn(
    message: AnyMessage,
) -> tuple[str, str] | None:
    """Convert a durable chat message to a generic memory turn.

    ``ToolMessage`` is intentionally ignored.  Tool output is operational
    context and may be re-derived by running a tool; persisting it as a user
    fact would make future memory updates depend on transient tool payloads.
    """
    if isinstance(message, HumanMessage):
        return "user", _content_to_text(message.content)
    if isinstance(message, AIMessage):
        return "assistant", _content_to_text(message.content)
    if isinstance(message, ToolMessage):
        return None
    if isinstance(message, SystemMessage):
        return None
    # Keep custom BaseMessage subclasses useful while avoiding arbitrary
    # framework metadata in durable memory.
    if isinstance(message, BaseMessage):
        role = "user" if message.type in {"human", "user"} else "assistant"
        return role, _content_to_text(message.content)
    return None


def _message_ids(messages: Sequence[AnyMessage]) -> list[str]:
    return [str(message.id) for message in messages]


class LangChainChatAdapter:
    """Run a ``langchain-core`` chat model with structured memory.

    ``chat_model`` is any object exposing ``invoke``/``ainvoke``.  It is
    optional so callers can use ``before_model`` and ``prepare_messages`` as a
    standalone history adapter.  The model is never asked to summarize or
    update memory; those operations are delegated to ``MemoryUpdater``.

    The adapter accepts and returns ordinary LangChain message sequences; it
    does not implement an agent-state or middleware protocol.
    """

    def __init__(
        self,
        memory: Memory,
        updater: MemoryUpdater,
        max_tokens: int,
        *,
        chat_model: Any | None = None,
        evict_fraction: float = 0.5,
        keep_messages: int = 20,
        max_memory_tokens: int | None = None,
        transcript: Transcript | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
        memory_selector: MemorySelector | None = None,
        update_verifier: MemoryUpdateVerifier | None = None,
        policy: StructuredMemoryPolicy | None = None,
        compactor: MemoryCompactor | None = None,
        compact_min_active_entries: int = 30,
        base_system_prompt: str = "",
    ) -> None:
        self.chat_model = chat_model
        self.memory = memory
        self.updater = updater
        self.policy = policy or memory.policy or updater.policy
        self.compactor = compactor
        self.compact_min_active_entries = compact_min_active_entries
        self.update_verifier = (
            update_verifier
            if update_verifier is not None
            else MemoryUpdateVerifier(policy=self.policy)
        )
        self.service = StructuredMemoryService(
            memory=memory,
            updater=updater,
            policy=self.policy,
            update_verifier=self.update_verifier,
            compactor=compactor,
            compact_min_active_entries=compact_min_active_entries,
        )
        self.max_tokens = max_tokens
        self.evict_fraction = evict_fraction
        self.keep_messages = max(1, keep_messages)
        self.max_memory_tokens = (
            max_memory_tokens if max_memory_tokens is not None else max_tokens // 2
        )
        self.transcript = transcript if transcript is not None else Transcript()
        self.token_counter = token_counter
        self.memory_selector = memory_selector or MemorySelector(
            token_estimator=_char_token_estimator,
            policy=self.policy,
        )
        self.base_system_prompt = base_system_prompt
        self._history: list[AnyMessage] = []
        self._turn_id_by_message_id: dict[str, int] = {}
        # A failed update retains this exact prefix for retry, even if a
        # caller adds more messages before invoking the next model call.
        self._pending_message_ids: tuple[str, ...] | None = None
        self._pending_turns: tuple[Turn, ...] = ()

    @property
    def messages(self) -> list[AnyMessage]:
        """Return the current model history (a copy, to protect invariants)."""
        return list(self._history)

    @property
    def history(self) -> list[AnyMessage]:
        """Alias for :attr:`messages` used by chat wrappers."""
        return self.messages

    @staticmethod
    def _ensure_ids(messages: Sequence[AnyMessage]) -> None:
        for message in messages:
            if message.id is None:
                message.id = str(uuid.uuid4())

    def _append_messages(self, messages: Sequence[AnyMessage]) -> None:
        """Append unseen messages, preserving caller-provided message IDs."""
        self._ensure_ids(messages)
        seen = {str(message.id) for message in self._history}
        for message in messages:
            if str(message.id) not in seen:
                self._history.append(message)
                seen.add(str(message.id))

    def set_messages(self, messages: Sequence[AnyMessage]) -> list[AnyMessage]:
        """Replace local history with ``messages`` and return a safe copy."""
        self._history = []
        self._append_messages(messages)
        return self.messages

    def append(self, message: AnyMessage) -> None:
        """Append one message to local history."""
        self._append_messages([message])

    @staticmethod
    def _pair_bounds(messages: Sequence[AnyMessage]) -> list[tuple[int, int]]:
        """Return ``[start, end)`` ranges for AI/tool protocol groups."""
        groups: list[tuple[int, int]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if isinstance(message, AIMessage) and message.tool_calls:
                call_ids = {
                    str(call.get("id"))
                    for call in message.tool_calls
                    if call.get("id") is not None
                }
                end = index + 1
                seen_ids: set[str] = set()
                while end < len(messages) and isinstance(messages[end], ToolMessage):
                    tool_id = getattr(messages[end], "tool_call_id", None)
                    if tool_id is None or not call_ids or str(tool_id) in call_ids:
                        seen_ids.add(str(tool_id) if tool_id is not None else "")
                        end += 1
                    else:
                        break
                # Include only actual responses when call IDs are available;
                # an AI message with no matching response is still one unit.
                if call_ids and seen_ids and not (seen_ids & call_ids):
                    end = index + 1
                groups.append((index, end))
                index = end
                continue
            if isinstance(message, ToolMessage):
                end = index + 1
                while end < len(messages) and isinstance(messages[end], ToolMessage):
                    end += 1
                groups.append((index, end))
                index = end
                continue
            index += 1
        return groups

    @classmethod
    def _find_safe_cutoff_point(
        cls, messages: Sequence[AnyMessage], cutoff_index: int
    ) -> int:
        """Snap a cutoff to an AI/tool protocol-group boundary.

        The return value is an exclusive prefix length.  Both directions are
        handled: a cutoff before a ``ToolMessage`` moves before its AI call,
        while a cutoff immediately after an AI call moves before that call.
        """
        cutoff_index = max(0, min(cutoff_index, len(messages)))
        for start, end in cls._pair_bounds(messages):
            if start < cutoff_index < end:
                return start
        # Never leave a preserved history beginning with a tool response.  An
        # orphan run is operationally one unit even if no AI call is present.
        while cutoff_index > 0 and cutoff_index < len(messages):
            if not isinstance(messages[cutoff_index], ToolMessage):
                break
            cutoff_index -= 1
        return cutoff_index

    def _find_cutoff(self, messages: Sequence[AnyMessage]) -> int:
        if not messages or self.token_counter(list(messages)) <= self.max_tokens:
            return 0
        target_tokens = max(1, int(self.max_tokens * (1 - self.evict_fraction)))
        left, right = 0, len(messages)
        candidate = len(messages)
        for _ in range(len(messages).bit_length() + 1):
            if left >= right:
                break
            middle = (left + right) // 2
            if self.token_counter(list(messages[middle:])) <= target_tokens:
                candidate = middle
                right = middle
            else:
                left = middle + 1
        if candidate == len(messages):
            candidate = left
        candidate = min(candidate, len(messages) - 1)
        candidate = min(candidate, len(messages) - min(self.keep_messages, len(messages)))
        return max(0, self._find_safe_cutoff_point(messages, candidate))

    def _mirror_durable_messages(
        self,
        messages: Sequence[AnyMessage],
    ) -> None:
        for message in messages:
            key = str(message.id)
            if key in self._turn_id_by_message_id:
                continue
            turn_data = _message_to_turn(message)
            if turn_data is None:
                continue
            role, content = turn_data
            appended = self.transcript.append(role, content)
            self._turn_id_by_message_id[key] = appended.id

    def _turns_for_ids(self, ids: Sequence[str]) -> list[Turn]:
        by_id = {turn.id: turn for turn in self.transcript.all()}
        return [
            by_id[self._turn_id_by_message_id[message_id]]
            for message_id in ids
            if message_id in self._turn_id_by_message_id
        ]

    def _pending_prefix(self) -> tuple[int, list[AnyMessage], list[Turn]] | None:
        if not self._pending_message_ids:
            return None
        ids = list(self._pending_message_ids)
        current = _message_ids(self._history)
        if current[: len(ids)] != ids:
            # A caller replaced/reordered history; the stale pending batch is
            # no longer safe to apply and can be recalculated.
            self._pending_message_ids = None
            self._pending_turns = ()
            return None
        return len(ids), list(self._history[: len(ids)]), list(self._pending_turns)

    def before_model(
        self,
        messages: Sequence[AnyMessage],
    ) -> list[AnyMessage]:
        """Evict a safe prefix and return the retained model history.

        Messages are removed from local history only after the updater batch
        commits successfully.
        """
        self.set_messages(list(messages))
        self._mirror_durable_messages(self._history)

        pending = self._pending_prefix()
        if pending is not None:
            cutoff, evicted, evicted_turns = pending
        else:
            cutoff = self._find_cutoff(self._history)
            evicted = list(self._history[:cutoff])
            evicted_ids = _message_ids(evicted)
            evicted_turns = self._turns_for_ids(evicted_ids)

        if cutoff <= 0:
            return self.messages

        # Do not send operational tool output to the updater.  A tool-only
        # prefix is safe to drop after the no-op transaction succeeds.
        if not evicted_turns:
            self._history = self._history[cutoff:]
            self._pending_message_ids = None
            self._pending_turns = ()
            return self.messages

        try:
            self.service.update_verifier = self.update_verifier
            update_result = self.service.update(evicted_turns)
        except Exception as exc:  # failure-safe: retain pending history
            logger.warning("LangChain chat memory update failed; retrying: %s", exc)
            self._pending_message_ids = tuple(_message_ids(evicted))
            self._pending_turns = tuple(evicted_turns)
            return self.messages

        if (
            not update_result.committed
            or update_result.rejected_ops
            or update_result.failure_reason == "verification_failed"
        ):
            logger.warning(
                "LangChain chat memory update was not committed; retaining pending messages: %s",
                update_result.failure_reason or update_result.rejected_ops,
            )
            self._pending_message_ids = tuple(_message_ids(evicted))
            self._pending_turns = tuple(evicted_turns)
            return self.messages

        self._history = self._history[cutoff:]
        self._pending_message_ids = None
        self._pending_turns = ()
        return self.messages

    @staticmethod
    def _query_from_messages(messages: Sequence[AnyMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return _content_to_text(message.content)
        return ""

    def _memory_system_prompt(self, system_prompt: str = "") -> str:
        query = self._query_from_messages(self._history)
        selected = self.memory_selector.select_for_answer(
            memory=self.memory,
            query=query,
            budget=self.max_memory_tokens,
        )
        context = build_answer_memory_context(memory=self.memory, entries=selected)
        rendered = context.rendered_context
        parts = [part for part in (self.base_system_prompt, system_prompt) if part]
        if rendered:
            parts.extend(("# Conversation Memory", rendered))
        return "\n\n".join(parts)

    def prepare_messages(
        self,
        messages: Sequence[AnyMessage] | None = None,
        *,
        system_prompt: str = "",
    ) -> list[AnyMessage]:
        """Return retained history with answer-time memory injection."""
        if messages is not None:
            self.set_messages(messages)
        prompt = self._memory_system_prompt(system_prompt)
        history = [message for message in self._history if not isinstance(message, SystemMessage)]
        return ([SystemMessage(content=prompt)] if prompt else []) + history

    def _maybe_compact(self) -> None:
        """Compatibility wrapper around framework-neutral compaction."""
        self.service.maybe_compact()

    def compaction_diagnostics(self) -> dict[str, Any]:
        """Return structured compaction diagnostics."""
        return self.service.compaction_diagnostics()

    def _invoke_model(self, messages: list[AnyMessage], **kwargs: Any) -> Any:
        if self.chat_model is None:
            raise RuntimeError("chat_model is required for invoke()")
        invoke = getattr(self.chat_model, "invoke", None)
        if invoke is None:
            if callable(self.chat_model):
                return self.chat_model(messages, **kwargs)
            raise TypeError("chat_model must expose invoke() or be callable")
        return invoke(messages, **kwargs)

    def invoke(
        self,
        message: str | BaseMessage | Sequence[AnyMessage],
        *,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> Any:
        """Append input, update memory if needed, invoke the chat model, and retain its reply."""
        if isinstance(message, str):
            self.append(HumanMessage(content=message))
        elif isinstance(message, BaseMessage):
            self.append(message)
        else:
            self._append_messages(list(message))
        self.before_model(self._history)
        response = self._invoke_model(self.prepare_messages(system_prompt=system_prompt), **kwargs)
        if isinstance(response, BaseMessage):
            self.append(response)
        return response

    async def ainvoke(
        self,
        message: str | BaseMessage | Sequence[AnyMessage],
        *,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> Any:
        """Async counterpart to :meth:`invoke` for Runnable chat models."""
        if isinstance(message, str):
            self.append(HumanMessage(content=message))
        elif isinstance(message, BaseMessage):
            self.append(message)
        else:
            self._append_messages(list(message))
        self.before_model(self._history)
        prepared = self.prepare_messages(system_prompt=system_prompt)
        if self.chat_model is None:
            raise RuntimeError("chat_model is required for ainvoke()")
        invoke = getattr(self.chat_model, "ainvoke", None)
        if invoke is None:
            result = self._invoke_model(prepared, **kwargs)
            if inspect.isawaitable(result):
                response = await result
            else:
                response = result
        else:
            response = await invoke(prepared, **kwargs)
        if isinstance(response, BaseMessage):
            self.append(response)
        return response


__all__ = [
    "LangChainChatAdapter",
]

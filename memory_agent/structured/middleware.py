"""LangChain middleware that replaces `SummarizationMiddleware` with structured memory.

This module is opt-in: it is never imported from `memory_agent/__init__.py`, so
the core package stays framework-free. Importing this module requires
`langchain` and `langgraph` to be installed.

The safe-cutoff search (`_find_safe_cutoff_point`) and the message-id /
replacement bookkeeping (`_ensure_ids`, `RemoveMessage(id=REMOVE_ALL_MESSAGES)`)
below are adapted from langchain's `SummarizationMiddleware`
(`langchain.agents.middleware.summarization`, langchain 1.3.11): never separate
an AIMessage carrying `tool_calls` from its `ToolMessage` responses, and give
every message a stable id before relying on the "remove all, then re-add"
reducer pattern.

Unlike that middleware, this one never inserts a summary message into the
conversation and never deletes messages on failure: if the memory update
fails or any op is rejected, `before_model` returns `None` and all messages
are kept for a retry on the next call. Memory only reaches the model via
system-prompt injection in `wrap_model_call`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from memory_agent.models.transcript import Turn
from memory_agent.models.policy import MemoryPolicy
from memory_agent.structured.memory import Memory
from memory_agent.structured.selector import MemorySelector
from memory_agent.structured.answer_context import (
    AnswerContextBudget,
    AnswerContextConfig,
    build_answer_memory_context,
)
from memory_agent.structured.transcript import Transcript
from memory_agent.structured.compactor import MemoryCompactor
from memory_agent.structured.updater import MemoryUpdater, UpdateFailed
from memory_agent.structured.verifier import MemoryUpdateVerifier

logger = logging.getLogger(__name__)

TokenCounter = Callable[[list[AnyMessage]], int]


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


def _message_to_turn_fields(message: AnyMessage) -> tuple[str, str]:
    """Map a LangChain message to (role, content) for a `Turn`.

    HumanMessage -> "user". AIMessage -> "assistant", with any tool_calls
    appended as `[tool_call] name(args)` lines so the updater can see what was
    attempted. ToolMessage -> "tool", content prefixed with the tool name when
    available. Anything else falls back to its `.type` and text content.
    """
    if isinstance(message, HumanMessage):
        return "user", _content_to_text(message.content)

    if isinstance(message, AIMessage):
        lines: list[str] = []
        text = _content_to_text(message.content)
        if text:
            lines.append(text)
        for tool_call in message.tool_calls or []:
            name = tool_call.get("name", "")
            args = tool_call.get("args", {})
            lines.append(f"[tool_call] {name}({args})")
        return "assistant", "\n".join(lines)

    if isinstance(message, ToolMessage):
        text = _content_to_text(message.content)
        name = getattr(message, "name", None)
        return "tool", f"[{name}] {text}" if name else text

    return getattr(message, "type", "unknown"), _content_to_text(getattr(message, "content", ""))


class StructuredMemoryMiddleware(AgentMiddleware):
    """Evicts old messages into structured memory instead of summarizing them.

    Borrows `SummarizationMiddleware`'s safe-cutoff and message-id mechanics
    (see module docstring) but hands evicted messages to a `MemoryUpdater`
    instead of an LLM summarization call, and never loses messages on
    failure. At least `keep_messages` recent messages are preserved verbatim.
    The rendered memory is injected into the system prompt on every model call.
    """

    def __init__(
        self,
        memory: Memory,
        updater: MemoryUpdater,
        max_tokens: int,
        evict_fraction: float = 0.5,
        keep_messages: int = 20,
        max_memory_tokens: int | None = None,
        max_tool_turn_chars: int | None = 2000,
        transcript: Transcript | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
        memory_selector: MemorySelector | None = None,
        update_verifier: MemoryUpdateVerifier | None = None,
        policy: MemoryPolicy | None = None,
        compactor: "MemoryCompactor | None" = None,
        compact_min_active_entries: int = 30,
    ) -> None:
        super().__init__()
        self.memory = memory
        self.updater = updater
        self.policy = policy or memory.policy or updater.policy
        # Optional subject-level compaction, triggered after successful
        # evictions once active entries exceed the threshold.
        self.compactor = compactor
        self.compact_min_active_entries = compact_min_active_entries
        self._last_compaction_failure_active: int | None = None
        self._compaction_retry_growth = 10
        # Semantic invariant check on updater output (defense-in-depth); pass
        # an explicit verifier to customize, or disable via a stub that always
        # passes. Default on: it only fires when the deterministic extraction
        # pipeline itself regressed.
        self.update_verifier = (
            update_verifier
            if update_verifier is not None
            else MemoryUpdateVerifier(policy=self.policy)
        )
        self.max_tokens = max_tokens
        self.evict_fraction = evict_fraction
        # A pre-existing clamp in _find_cutoff already guarantees at least one
        # message survives eviction; clamp here so the configured value never
        # silently diverges from the effective floor.
        self.keep_messages = max(1, keep_messages)
        self.max_memory_tokens = (
            max_memory_tokens if max_memory_tokens is not None else max_tokens // 2
        )
        self.max_tool_turn_chars = max_tool_turn_chars
        self.transcript = transcript if transcript is not None else Transcript()
        self.token_counter = token_counter
        self.memory_selector = memory_selector or MemorySelector(
            token_estimator=_char_token_estimator,
            policy=self.policy,
        )
        self._turn_id_by_message_id: dict[str, int] = {}

    @staticmethod
    def _ensure_ids(messages: list[AnyMessage]) -> None:
        # Adapted from SummarizationMiddleware._ensure_message_ids (langchain
        # 1.3.11): every message needs a stable id for the add_messages
        # reducer and for our own mirrored-turn bookkeeping.
        for message in messages:
            if message.id is None:
                message.id = str(uuid.uuid4())

    def _mirror_messages(self, messages: list[AnyMessage]) -> None:
        """Mirror not-yet-seen messages into the transcript. Idempotent.

        Tool output is deterministically bounded before updater-LLM extraction.
        Tool results are re-derivable by re-running the tool, so sending huge
        outputs to the updater wastes tokens and risks invented details from
        paraphrase or truncation.
        """
        for message in messages:
            if message.id in self._turn_id_by_message_id:
                continue
            role, content = _message_to_turn_fields(message)
            if (
                role == "tool"
                and self.max_tool_turn_chars is not None
                and len(content) > self.max_tool_turn_chars
            ):
                content = (
                    content[: self.max_tool_turn_chars]
                    + "\n[tool output truncated before memory extraction; "
                    "re-run the tool for the full output]"
                )
            turn = self.transcript.append(role, content)
            self._turn_id_by_message_id[message.id] = turn.id

    def _turns_for(self, messages: list[AnyMessage]) -> list[Turn]:
        turns_by_id = {turn.id: turn for turn in self.transcript.all()}
        return [turns_by_id[self._turn_id_by_message_id[message.id]] for message in messages]

    @staticmethod
    def _find_safe_cutoff_point(messages: list[AnyMessage], cutoff_index: int) -> int:
        """Find a safe cutoff point that doesn't split AI/Tool message pairs.

        Adapted verbatim in spirit from
        `SummarizationMiddleware._find_safe_cutoff_point` (langchain 1.3.11):
        if the message at `cutoff_index` is a `ToolMessage`, search backward
        for the `AIMessage` that issued the corresponding tool call and move
        the cutoff there so the pair stays together. Falls back to advancing
        past the ToolMessages if no matching AIMessage is found.
        """
        if cutoff_index >= len(messages) or not isinstance(messages[cutoff_index], ToolMessage):
            return cutoff_index

        tool_call_ids: set[str] = set()
        idx = cutoff_index
        while idx < len(messages) and isinstance(messages[idx], ToolMessage):
            tool_message = messages[idx]
            if tool_message.tool_call_id:
                tool_call_ids.add(tool_message.tool_call_id)
            idx += 1

        for i in range(cutoff_index - 1, -1, -1):
            message = messages[i]
            if isinstance(message, AIMessage) and message.tool_calls:
                ai_tool_call_ids = {tc.get("id") for tc in message.tool_calls if tc.get("id")}
                if tool_call_ids & ai_tool_call_ids:
                    return i

        return idx

    def _find_cutoff(self, messages: list[AnyMessage]) -> int:
        """Choose the largest safe eviction cutoff that meets the eviction budget.

        Uses the same binary-search-for-a-token-budget shape as
        `SummarizationMiddleware._find_token_based_cutoff`, targeting
        `max_tokens * (1 - evict_fraction)` tokens remaining, then snaps to a
        safe cutoff via `_find_safe_cutoff_point`. Always leaves at least
        `keep_messages` recent messages preserved when that many exist.
        """
        if not messages:
            return 0

        target_tokens = int(self.max_tokens * (1 - self.evict_fraction))
        if target_tokens <= 0:
            target_tokens = 1

        if self.token_counter(messages) <= target_tokens:
            return 0

        left, right = 0, len(messages)
        cutoff_candidate = len(messages)
        max_iterations = len(messages).bit_length() + 1
        for _ in range(max_iterations):
            if left >= right:
                break
            mid = (left + right) // 2
            if self.token_counter(messages[mid:]) <= target_tokens:
                cutoff_candidate = mid
                right = mid
            else:
                left = mid + 1

        if cutoff_candidate == len(messages):
            cutoff_candidate = left
        if cutoff_candidate >= len(messages):
            cutoff_candidate = len(messages) - 1 if len(messages) > 1 else 0

        min_preserved = min(self.keep_messages, len(messages))
        max_cutoff = len(messages) - min_preserved
        cutoff_candidate = min(cutoff_candidate, max_cutoff)
        if cutoff_candidate <= 0:
            return 0

        safe_cutoff = self._find_safe_cutoff_point(messages, cutoff_candidate)
        if safe_cutoff > max_cutoff:
            # The fallback branch of _find_safe_cutoff_point advanced past a
            # run of orphaned ToolMessages; respect the configured tail floor
            # instead by walking back to the nearest non-ToolMessage.
            safe_cutoff = max_cutoff
            while safe_cutoff > 0 and isinstance(messages[safe_cutoff], ToolMessage):
                safe_cutoff -= 1

        return max(safe_cutoff, 0)

    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        """Evict old messages into structured memory when over the token budget."""
        messages: list[AnyMessage] = state["messages"]
        self._ensure_ids(messages)
        self._mirror_messages(messages)

        if self.token_counter(messages) <= self.max_tokens:
            return None

        cutoff = self._find_cutoff(messages)
        if cutoff <= 0:
            return None

        evicted = messages[:cutoff]
        preserved = messages[cutoff:]
        evicted_turns = self._turns_for(evicted)

        try:
            _applied, rejected = self.updater.update(self.memory, evicted_turns)
        except UpdateFailed as exc:
            logger.warning("Memory update failed; keeping messages for retry: %s", exc)
            return None

        if rejected:
            logger.warning(
                "Memory updater produced rejected ops; keeping messages for retry: %s", rejected
            )
            return None

        verification = self.update_verifier.verify(
            evicted_turns=evicted_turns,
            applied_ops=_applied,
            rejected_ops=rejected,
            memory=self.memory,
        )
        if not verification.passed:
            logger.warning(
                "Memory update failed semantic verification; keeping messages for retry: %s",
                verification.errors,
            )
            return None

        self._maybe_compact()

        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *preserved]}

    def _maybe_compact(self) -> None:
        """Consolidate same-subject entries once active memory grows too large.

        Runs only after a fully validated eviction, and never blocks it: the
        evicted turns are already safely in memory, so a compaction failure
        just means memory stays more granular until the next attempt.
        """
        if self.compactor is None:
            return
        active = sum(
            entry.status == "active" for entry in self.memory.entries.values()
        )
        if active <= self.compact_min_active_entries:
            self.compactor.record_skip("below_threshold")
            return
        if (
            self._last_compaction_failure_active is not None
            and active < self._last_compaction_failure_active + self._compaction_retry_growth
        ):
            self.compactor.record_skip("circuit_breaker")
            return
        try:
            applied, rejected = self.compactor.compact(self.memory)
        except UpdateFailed as exc:
            self._last_compaction_failure_active = active
            logger.warning("Memory compaction failed; continuing uncompacted: %s", exc)
            return
        if rejected:
            self._last_compaction_failure_active = active
            logger.warning("Memory compaction ops rejected; continuing uncompacted: %s", rejected)
        elif applied:
            self._last_compaction_failure_active = None
            logger.info("Memory compaction applied %d ops", len(applied))

    async def abefore_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        """Async wrapper: the eviction logic above does no I/O of its own."""
        return self.before_model(state, runtime)

    @staticmethod
    def _query_from_messages(messages: list[AnyMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return _content_to_text(message.content)
        return ""

    def wrap_model_call(self, request: "ModelRequest[Any]", handler: Callable[..., Any]) -> Any:
        """Inject the rendered structured memory into the system prompt.

        The token budget is enforced once, by the selector. Rendering must not
        re-apply `max_tokens`: `Memory.render` has no notion of pinned
        sections, so a second budget pass would omit exactly the pinned
        entries the selector guarantees.
        """
        query = self._query_from_messages(list(request.messages))
        answer_context = build_answer_memory_context(
            query=query,
            memory=self.memory,
            config=AnswerContextConfig(selector=self.memory_selector),
            budget=AnswerContextBudget(max_tokens=self.max_memory_tokens),
        )
        rendered = answer_context.rendered_context
        if rendered:
            new_prompt = (request.system_prompt or "") + "\n\n# Conversation Memory\n" + rendered
            if hasattr(request, "override"):
                request = request.override(system_prompt=new_prompt)
            else:
                request.system_prompt = new_prompt
        return handler(request)

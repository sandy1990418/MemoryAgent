"""Run one BEAM chat case answered by a deepagents agent with agentic retrieval.

This is the `create_deep_agent` variant of `scripts/run_beam_case.py`. The
ingestion and StructuredMemoryMiddleware stages are identical, but the
answering stage differs: instead of one `OpenAIClient.complete` call with
pre-retrieved mem0 top-k context injected into the prompt, each probing
question is handed to a deepagents agent that must perform its own retrieval
by calling a `search_long_term_memory` tool (possibly several times with
different queries) before answering.

Requires Python >= 3.11 (a `deepagents` constraint). The project's main
`.venv` is Python 3.10, so run this script with the dedicated interpreter:

    .venv-deepagents/bin/pip install -r requirements-deepagents.txt
    .venv-deepagents/bin/python scripts/run_beam_case_deepagent.py

`deepagents` is imported lazily inside `build_answer_agent`, so importing this
module (for its pure helpers) works on Python 3.10 as well.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage

from memory_agent import AGENT_SECTIONS, Mem0LongTermMemory, Memory, MemoryUpdater, OpenAIClient
from memory_agent.models.beam import BeamDeepAgentRunConfig
from memory_agent.structured.middleware import StructuredMemoryMiddleware, _content_to_text
from scripts.run_beam_case import (
    DEFAULT_CHAT_PATH,
    DEFAULT_PROBES_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TOPICS_PATH,
    apply_message_update,
    build_context,
    flatten_chunks,
    flatten_message_batches,
    judge_response,
    load_json,
    load_topic,
    reference_answer,
    render_message_tail,
    rubric_hit,
    structured_memory_stats,
)

SEARCH_TOOL_NAME = "search_long_term_memory"


def final_ai_text(messages: list[AnyMessage]) -> str:
    """Return the text of the last AIMessage that is not a tool-call step."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = _content_to_text(message.content).strip()
            if text:
                return text
    return ""


def collect_tool_trace(
    messages: list[AnyMessage], tool_name: str = SEARCH_TOOL_NAME
) -> list[dict[str, Any]]:
    """Collect the retrieval tool calls the agent issued, in order."""
    trace: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls or []:
            if tool_call.get("name") == tool_name:
                trace.append({"name": tool_call.get("name"), "args": tool_call.get("args", {})})
    return trace


def make_search_tool(
    memory: Mem0LongTermMemory,
    user_id: str,
    default_limit: int,
    max_hit_chars: int,
):
    def search_long_term_memory(query: str, limit: int = 8) -> str:
        """Search the long-term memory store of the BEAM conversation.

        Returns the most relevant stored transcript excerpts for `query`,
        formatted as text blocks tagged with their source chat ids. Call this
        tool before answering; call it again with reworded or narrower
        queries if the first results do not cover the question.
        """
        use_limit = limit if limit > 0 else default_limit
        hits = memory.search(query, user_id=user_id, limit=use_limit)
        return build_context(hits, max_hit_chars=max_hit_chars)

    return search_long_term_memory


def build_agent_system_prompt(
    structured_middleware: StructuredMemoryMiddleware | None,
    active_messages: list[AnyMessage],
    structured_answer_tokens: int,
    max_active_context_chars: int,
) -> str:
    if structured_middleware is None:
        conversation_memory = "(StructuredMemoryMiddleware was not used.)"
        working_tail = "(No structured working-context tail.)"
    else:
        conversation_memory = structured_middleware.memory.render(
            max_tokens=structured_answer_tokens,
            include_superseded=True,
        ) or "(No structured memory entries.)"
        working_tail = render_message_tail(active_messages, max_active_context_chars)

    return (
        "You answer BEAM long-term-memory probing questions.\n"
        f"Always call the {SEARCH_TOOL_NAME} tool to gather supporting recall "
        "before answering. First decompose the question into its distinct "
        "facets (sub-topics, features, time periods, or claimed items), then "
        "issue one search per facet: at least 2 searches for any question, and "
        "at least 4 for comprehensive summary or multi-part questions (pass "
        "limit=12 for broad questions). Use only the # Conversation "
        "Memory and # Working Conversation Tail sections below plus what the tool "
        "returns. The # Conversation Memory section is the structured current "
        "state produced by StructuredMemoryMiddleware; prefer it if it conflicts "
        "with tool results. If the answer is not supported by the available "
        "context, say that the provided chat does not contain enough information. "
        "Do not infer background, previous projects, user feedback, causes, or "
        "outcomes from nearby project facts; those details must be explicitly "
        "supported. When a question asks about something never explicitly "
        "discussed (personal background, earlier projects, feedback details), "
        "composing an answer from adjacent or current-project facts is wrong: "
        "abstain and say the information is not in the chat. "
        "For contradiction questions, if memory contains both a "
        "denial/correction and an affirmative project fact, explicitly state "
        "that there is contradictory information and ask which statement is "
        "correct. For contradiction_resolution questions specifically, always "
        "issue at least two searches before answering: one for the topic as "
        "asked, and a second using negation phrasing (e.g., 'never', 'have not', "
        "'not yet', 'no longer') to check whether the user also denied, "
        "corrected, or reversed that claim elsewhere. Only conclude there is no "
        "contradiction after both searches turn up consistent information. "
        "For ordering or temporal questions, answer from the # Chronological "
        "Order section in the question message: it reflects the order topics "
        "were actually first raised. Do not answer from planned schedules or "
        "phase plans found in timeline entries. Use turn ids, chat ids, "
        "and provenance to verify chronology, and obey any requested item "
        "count exactly by grouping related work into exactly that many items. "
        "For 'how many' questions, search each candidate item "
        "separately, enumerate every distinct mention across sessions, and "
        "state the total count explicitly at the start of the answer. When "
        "retrieved evidence contains exact identifiers - version numbers, "
        "algorithm or library names, error messages, tool or platform names, "
        "dates - repeat them verbatim in the answer instead of paraphrasing. "
        "Keep answers concise and do not include tutorials or "
        "extra implementation advice - except for comprehensive summary "
        "questions, where completeness beats brevity: enumerate every distinct "
        "incident, error resolved, tool adopted, and enhancement found in "
        "retrieval, naming exact libraries, algorithms, error types, and "
        "platforms verbatim rather than generalizing them away. Follow any "
        "remembered user instructions "
        "that are relevant to the question.\n\n"
        "# Conversation Memory\n"
        f"{conversation_memory}\n\n"
        "# Working Conversation Tail\n"
        "Recent active messages not yet evicted into structured memory.\n"
        f"{working_tail}"
    )


def build_answer_agent(
    model: str,
    search_tool: Any,
    system_prompt: str,
) -> Any:
    """Build a stateless deepagents agent (no checkpointer: each `invoke`
    starts fresh, so probing questions cannot contaminate each other)."""
    from deepagents import create_deep_agent  # lazy import; needs Python >= 3.11

    model_spec = model if ":" in model else f"openai:{model}"
    return create_deep_agent(
        model=model_spec,
        tools=[search_tool],
        system_prompt=system_prompt,
    )


def ask_agent(
    agent: Any,
    topic: dict[str, Any],
    question_type: str,
    question: str,
    recursion_limit: int,
    structured_middleware: StructuredMemoryMiddleware | None = None,
    structured_answer_tokens: int = 4000,
) -> tuple[str, list[dict[str, Any]], int]:
    topic_text = json.dumps(topic, ensure_ascii=False)
    if structured_middleware is None:
        relevant_memory = "(StructuredMemoryMiddleware was not used.)"
        chronological = "(StructuredMemoryMiddleware was not used.)"
        denials = "(StructuredMemoryMiddleware was not used.)"
    else:
        selected_entries = structured_middleware.memory_selector.select(
            memory=structured_middleware.memory,
            query=question,
            max_tokens=structured_answer_tokens,
        )
        relevant_memory = (
            structured_middleware.memory.render(entries=selected_entries)
            or "(No relevant structured memory entries.)"
        )
        chronological = (
            structured_middleware.memory.render_chronological(
                max_tokens=structured_answer_tokens // 2,
                # Identifier entries (versions, dates, paths) drown out the
                # topical mention-order signal ordering questions need.
                exclude_sections={"exact_values"},
            )
            or "(No chronological memory entries.)"
        )
        # A small answering model reliably misses a lone status_changes entry
        # buried among a hundred others, then trusts affirmative retrieval
        # hits instead. Surface denials/corrections in their own block right
        # next to the question so the conflict cannot be overlooked.
        denial_entries = [
            entry
            for entry in structured_middleware.memory.entries.values()
            if entry.section == "status_changes" and entry.status == "active"
        ]
        denials = (
            structured_middleware.memory.render(entries=denial_entries)
            or "(No recorded denials or corrections.)"
        )
    user = (
        f"Topic metadata:\n{topic_text}\n\n"
        f"Question type: {question_type}\n\n"
        "# Question-Relevant Structured Memory\n"
        f"{relevant_memory}\n\n"
        "# Chronological Order\n"
        "Entries ordered by when they were first mentioned (earliest first); use this "
        "to answer ordering or sequence questions.\n"
        f"{chronological}\n\n"
        "# Recorded Denials and Corrections\n"
        "Explicit user statements denying, correcting, or reversing earlier claims. "
        "Before answering, check whether any entry below is about the same subject "
        "as the question. If one conflicts with affirmative facts from memory or "
        "retrieval, you MUST state there is contradictory information, quote both "
        "sides, and ask which statement is correct - do not silently pick one side.\n"
        f"{denials}\n\n"
        f"Probing question:\n{question}\n\n"
        f"Use the {SEARCH_TOOL_NAME} tool to retrieve supporting evidence "
        "before you answer: decompose the question into facets and issue one "
        "search per facet (minimum 2 searches; at least 4 for comprehensive "
        "summary questions). If the retrieved evidence is only topically related "
        "but does not directly answer the question, say the provided chat does "
        "not contain enough information."
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user}]},
        config={"recursion_limit": recursion_limit},
    )
    messages = result["messages"]
    return final_ai_text(messages), collect_tool_trace(messages), len(messages)


def run(args: argparse.Namespace | BeamDeepAgentRunConfig) -> dict[str, Any]:
    if isinstance(args, argparse.Namespace):
        args = BeamDeepAgentRunConfig.from_args(args)

    load_dotenv(args.env_file)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; add it to .env or the environment.")

    memory_mode = f"{args.memory_mode}_deepagent"
    run_id = time.strftime("%Y%m%d-%H%M%S")
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    store_dir = args.store_dir or results_dir / f"mem0_store_{run_id}"
    output_path = args.output or results_dir / f"memory_agent_{memory_mode}_results_{run_id}.json"

    os.environ.setdefault("MEM0_DIR", str(store_dir))
    os.environ.setdefault("MEM0_TELEMETRY", "False")

    chat = load_json(args.chat)
    probes = load_json(args.probes)
    topics = load_json(args.topics)
    topic = load_topic(topics)
    chunks = flatten_chunks(chat)
    message_batches = flatten_message_batches(chat)

    memory = Mem0LongTermMemory.from_local(
        data_dir=str(store_dir),
        collection_name="beam_100k_case_1",
        llm_model=args.mem0_llm_model,
        infer=False,
    )

    if args.skip_ingest:
        print(f"Skipping ingestion and reusing {store_dir}", flush=True)
    else:
        print(f"Ingesting {len(chunks)} raw BEAM chunks into {store_dir}", flush=True)
        for index, chunk in enumerate(chunks, start=1):
            memory.add(
                [{"role": "user", "content": chunk.text}],
                user_id=args.user_id,
                metadata=chunk.metadata,
            )
            if index % 25 == 0 or index == len(chunks):
                print(f"  ingested {index}/{len(chunks)}", flush=True)

    structured_middleware: StructuredMemoryMiddleware | None = None
    active_messages: list[AnyMessage] = []
    if args.memory_mode == "structured_mem0":
        structured_middleware = StructuredMemoryMiddleware(
            memory=Memory(sections=AGENT_SECTIONS),
            updater=MemoryUpdater(
                llm=OpenAIClient(args.structured_model),
                sections=AGENT_SECTIONS,
            ),
            max_tokens=args.structured_max_tokens,
            evict_fraction=args.structured_evict_fraction,
            keep_messages=args.structured_keep_messages,
            max_memory_tokens=args.structured_max_memory_tokens,
        )

        print(
            "Processing BEAM transcript with StructuredMemoryMiddleware "
            f"({len(message_batches)} user/assistant pair batches)",
            flush=True,
        )
        for index, batch in enumerate(message_batches, start=1):
            active_messages.extend(batch)
            update = structured_middleware.before_model({"messages": active_messages}, None)
            active_messages = apply_message_update(active_messages, update)
            if index % 25 == 0 or index == len(message_batches):
                print(
                    f"  structured processed {index}/{len(message_batches)}; "
                    f"active_messages={len(active_messages)}; "
                    f"entries={len(structured_middleware.memory.entries)}",
                    flush=True,
                )

        if args.structured_flush_final and active_messages:
            old_max_tokens = structured_middleware.max_tokens
            structured_middleware.max_tokens = 1
            update = structured_middleware.before_model({"messages": active_messages}, None)
            active_messages = apply_message_update(active_messages, update)
            structured_middleware.max_tokens = old_max_tokens
            print(
                "  structured final flush; "
                f"active_messages={len(active_messages)}; "
                f"entries={len(structured_middleware.memory.entries)}",
                flush=True,
            )

    search_tool = make_search_tool(
        memory=memory,
        user_id=args.user_id,
        default_limit=args.top_k,
        max_hit_chars=args.max_hit_chars,
    )
    agent = build_answer_agent(
        model=args.answer_model,
        search_tool=search_tool,
        system_prompt=build_agent_system_prompt(
            structured_middleware=structured_middleware,
            active_messages=active_messages,
            structured_answer_tokens=args.structured_answer_tokens,
            max_active_context_chars=args.max_active_context_chars,
        ),
    )
    judge_llm = OpenAIClient(args.judge_model) if args.judge_model else None

    output: dict[str, Any] = {
        "run_id": run_id,
        "memory_mode": memory_mode,
        "chat": str(args.chat),
        "probes": str(args.probes),
        "topics": str(args.topics),
        "store_dir": str(store_dir),
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "structured_model": args.structured_model if structured_middleware is not None else None,
        "search_tool_default_limit": args.top_k,
        "recursion_limit": args.recursion_limit,
        "topic": topic,
        "structured_memory": (
            structured_middleware.memory.render(include_superseded=True)
            if structured_middleware is not None
            else None
        ),
        "structured_transcript_length": (
            len(structured_middleware.transcript) if structured_middleware is not None else None
        ),
        "structured_active_messages": len(active_messages) if structured_middleware is not None else None,
        "results": {},
        "summary": {},
    }

    total_hits = 0
    total_rubrics = 0
    total_tool_calls = 0
    total_judge_hits = 0
    total_judge_rubrics = 0
    for question_type, items in probes.items():
        print(f"Answering {question_type}: {len(items)} question(s)", flush=True)
        category_results = []
        category_hits = 0
        category_rubrics = 0
        category_judge_hits = 0
        category_judge_rubrics = 0

        for item_index, item in enumerate(items):
            question = item["question"]
            response, tool_trace, message_count = ask_agent(
                agent=agent,
                topic=topic,
                question_type=question_type,
                question=question,
                recursion_limit=args.recursion_limit,
                structured_middleware=structured_middleware,
                structured_answer_tokens=args.structured_answer_tokens,
            )
            total_tool_calls += len(tool_trace)
            rubric_checks = [rubric_hit(response, line) for line in item.get("rubric", [])]
            hit_count = sum(1 for check in rubric_checks if check["hit"])
            category_hits += hit_count
            category_rubrics += len(rubric_checks)

            judge_checks = []
            judge_hit_count = 0
            if judge_llm is not None:
                judge_checks = judge_response(
                    llm=judge_llm,
                    model=args.judge_model,
                    question_type=question_type,
                    question=question,
                    reference=reference_answer(item),
                    response=response,
                    rubric_lines=list(item.get("rubric", [])),
                )
                judge_hit_count = sum(1 for check in judge_checks if check["passed"])
                category_judge_hits += judge_hit_count
                category_judge_rubrics += len(judge_checks)

            print(
                f"  {question_type}[{item_index}] "
                f"heuristic={hit_count}/{len(rubric_checks)} "
                f"searches={len(tool_trace)}"
                + (
                    f" judge={judge_hit_count}/{len(judge_checks)}"
                    if judge_llm is not None
                    else ""
                ),
                flush=True,
            )
            category_results.append(
                {
                    "question": question,
                    "reference_answer": reference_answer(item),
                    "llm_response": response,
                    "tool_trace": tool_trace,
                    "tool_calls": len(tool_trace),
                    "agent_messages": message_count,
                    "structured_memory_used": structured_middleware is not None,
                    "rubric_checks": rubric_checks,
                    "heuristic_rubric_hits": hit_count,
                    "heuristic_rubric_total": len(rubric_checks),
                    "judge_checks": judge_checks if judge_llm is not None else None,
                    "judge_rubric_hits": judge_hit_count if judge_llm is not None else None,
                    "judge_rubric_total": len(judge_checks) if judge_llm is not None else None,
                }
            )

        total_hits += category_hits
        total_rubrics += category_rubrics
        total_judge_hits += category_judge_hits
        total_judge_rubrics += category_judge_rubrics
        output["results"][question_type] = category_results
        output["summary"][question_type] = {
            "heuristic_rubric_hits": category_hits,
            "heuristic_rubric_total": category_rubrics,
            "heuristic_rubric_rate": round(category_hits / category_rubrics, 3)
            if category_rubrics
            else None,
            "judge_rubric_hits": category_judge_hits if judge_llm is not None else None,
            "judge_rubric_total": category_judge_rubrics if judge_llm is not None else None,
            "judge_rubric_rate": round(category_judge_hits / category_judge_rubrics, 3)
            if judge_llm is not None and category_judge_rubrics
            else None,
        }

    output["summary"]["overall"] = {
        "chunks_ingested": len(chunks),
        "structured_entries": (
            len(structured_middleware.memory.entries) if structured_middleware is not None else 0
        ),
        "structured_transcript_length": (
            len(structured_middleware.transcript) if structured_middleware is not None else 0
        ),
        "structured_active_messages": len(active_messages) if structured_middleware is not None else 0,
        "questions_answered": sum(len(items) for items in probes.values()),
        "search_tool_calls": total_tool_calls,
        "structured_memory_stats": structured_memory_stats(
            structured_middleware.memory if structured_middleware is not None else None
        ),
        "heuristic_rubric_hits": total_hits,
        "heuristic_rubric_total": total_rubrics,
        "heuristic_rubric_rate": round(total_hits / total_rubrics, 3) if total_rubrics else None,
        "judge_rubric_hits": total_judge_hits if judge_llm is not None else None,
        "judge_rubric_total": total_judge_rubrics if judge_llm is not None else None,
        "judge_rubric_rate": round(total_judge_hits / total_judge_rubrics, 3)
        if judge_llm is not None and total_judge_rubrics
        else None,
        "note": (
            "Heuristic plus local LLM-as-judge over rubrics."
            if judge_llm is not None
            else "Heuristic only. Pass --judge-model or set BEAM_JUDGE_MODEL to run LLM-as-judge."
        ),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote results to {output_path}")
    print(json.dumps(output["summary"]["overall"], indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat", type=Path, default=DEFAULT_CHAT_PATH)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES_PATH)
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--user-id", default="beam-100k-case-1")
    parser.add_argument(
        "--memory-mode",
        choices=("structured_mem0", "raw_mem0"),
        default="structured_mem0",
        help="structured_mem0 runs StructuredMemoryMiddleware before answering; raw_mem0 relies on the agent's tool searches only.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-hit-chars", type=int, default=6000)
    parser.add_argument("--max-active-context-chars", type=int, default=12000)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--answer-model", default=os.getenv("BEAM_ANSWER_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--structured-model",
        default=os.getenv("BEAM_MEMORY_MODEL", os.getenv("MEMORY_MODEL", "gpt-4o-mini")),
    )
    parser.add_argument("--structured-max-tokens", type=int, default=12000)
    parser.add_argument("--structured-max-memory-tokens", type=int, default=3000)
    parser.add_argument("--structured-answer-tokens", type=int, default=4000)
    parser.add_argument("--structured-evict-fraction", type=float, default=0.5)
    parser.add_argument("--structured-keep-messages", type=int, default=2)
    parser.add_argument("--no-structured-flush-final", dest="structured_flush_final", action="store_false")
    parser.set_defaults(structured_flush_final=True)
    parser.add_argument("--mem0-llm-model", default=os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--recursion-limit", type=int, default=50)
    parser.add_argument("--judge-model", default=os.getenv("BEAM_JUDGE_MODEL") or None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

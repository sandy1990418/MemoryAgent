"""Run one BEAM chat case answered by an optional DeepAgent baseline.

This is the `create_deep_agent` comparison variant of
`scripts/run_beam_case.py`. It uses the same public chat-memory snapshot and
exposes a selector-backed search tool to the optional DeepAgent dependency.

Requires Python >= 3.11 (a `deepagents` constraint). The project's main
`.venv` is Python 3.10, so run this script with the dedicated interpreter:

    .venv-deepagents/bin/pip install -r requirements-deepagents.txt
    .venv-deepagents/bin/python scripts/run_beam_case_deepagent.py

`deepagents` is imported lazily inside `build_answer_agent`, so importing this
module (for its pure helpers) works on Python 3.10 as well.
"""

from __future__ import annotations

# ruff: noqa: E402

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

from memory_agent.clients.llm import OpenAIClient, TokenLedger
from memory_agent.clients.llm import LangChainTokenCallback
from memory_agent.application.chat import ChatMemory
from scripts.beam_models import (
    BeamDeepAgentRunConfig,
    beam_config_from_argv,
)
from memory_agent.models.config import product_config_from_argv
from memory_agent.adapters.langchain.structured_memory import _content_to_text
from scripts.run_beam_case import (
    DEFAULT_RESULTS_DIR,
    beam_answers_from_results,
    beam_evaluation_from_results,
    build_structured_beam_middleware,
    update_chat_memory,
    chat_batch_chars,
    memory_selector_for,
    default_answers_output_path,
    default_evaluation_output_path,
    beam_config_snapshot,
    current_source_commit,
    current_source_state,
    flatten_chunks,
    flatten_message_batches,
    judge_response,
    judge_score,
    load_json,
    load_topic,
    reference_answer,
    render_message_tail,
    rubric_hit,
    select_probes,
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
    memory: Any,
    default_limit: int,
    max_hit_chars: int,
):
    def search_long_term_memory(query: str, limit: int = 8) -> str:
        """Search the public chat-memory store of the BEAM conversation.

        Returns the most relevant stored transcript excerpts for `query`,
        formatted as text blocks tagged with their source chat ids. Call this
        tool before answering; call it again with reworded or narrower
        queries if the first results do not cover the question.
        """
        use_limit = limit if limit > 0 else default_limit
        selector = memory_selector_for(memory)
        entries = selector.select(memory=memory.memory, query=query, max_tokens=max(1, use_limit * 100))
        rendered = memory.memory.render(entries=entries) or "No matching chat memory."
        return rendered[:max_hit_chars] if max_hit_chars > 0 else rendered

    return search_long_term_memory


def build_agent_system_prompt(
    structured_middleware: ChatMemory | Any | None,
    active_messages: list[AnyMessage],
    structured_answer_tokens: int,
    max_active_context_chars: int,
    retrieval_enabled: bool = True,
) -> str:
    if structured_middleware is None:
        conversation_memory = "(ChatMemory was not used.)"
        working_tail = "(No chat working-context tail.)"
    else:
        conversation_memory = structured_middleware.memory.render(
            max_tokens=structured_answer_tokens,
            include_superseded=True,
        ) or "(No structured memory entries.)"
        working_tail = render_message_tail(active_messages, max_active_context_chars)

    if retrieval_enabled:
        retrieval_protocol = (
            f"Use the {SEARCH_TOOL_NAME} tool when the prompt memory does not "
            "contain enough direct evidence. For broad or multi-part questions, "
            "use a few targeted searches. "
        )
    else:
        retrieval_protocol = (
            "No retrieval tool is available. Answer using only the memory "
            "sections in this prompt and in the question message. "
        )

    return (
        "You answer long-term-memory questions.\n"
        + retrieval_protocol
        + "Use only available memory and, when a tool is available, retrieved "
        "evidence. Be concise. If the evidence is insufficient, say so. If the "
        "evidence conflicts, mention the conflict instead of choosing silently. "
        "Use chronological evidence for ordering or time questions. Follow "
        "relevant remembered user preferences.\n\n"
        "# Conversation Memory\n"
        f"{conversation_memory}\n\n"
        "# Working Conversation Tail\n"
        "Recent messages not yet folded into memory.\n"
        f"{working_tail}"
    )


def build_answer_agent(
    model: str,
    search_tool: Any | None,
    system_prompt: str,
) -> Any:
    """Build a stateless deepagents agent (no checkpointer: each `invoke`
    starts fresh, so probing questions cannot contaminate each other).
    `search_tool=None` builds a retrieval-free agent (structured_only mode)."""
    from deepagents import create_deep_agent  # lazy import; needs Python >= 3.11

    model_spec = model if ":" in model else f"openai:{model}"
    return create_deep_agent(
        model=model_spec,
        tools=[search_tool] if search_tool is not None else [],
        system_prompt=system_prompt,
    )


def ask_agent(
    agent: Any,
    topic: dict[str, Any],
    question_type: str,
    question: str,
    recursion_limit: int,
    structured_middleware: ChatMemory | Any | None = None,
    structured_answer_tokens: int = 4000,
    retrieval_enabled: bool = True,
    token_ledger: TokenLedger | None = None,
) -> tuple[str, list[dict[str, Any]], int]:
    topic_text = json.dumps(topic, ensure_ascii=False)
    if structured_middleware is None:
        relevant_memory = "(ChatMemory was not used.)"
        chronological = "(ChatMemory was not used.)"
        denials = "(ChatMemory was not used.)"
    else:
        selected_entries = memory_selector_for(structured_middleware).select(
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
        "Entries ordered by first mention, earliest first.\n"
        f"{chronological}\n\n"
        "# Recorded Denials and Corrections\n"
        "Explicit user statements denying, correcting, or reversing earlier claims.\n"
        f"{denials}\n\n"
        f"Probing question:\n{question}\n\n"
        + (
            f"Use the {SEARCH_TOOL_NAME} tool if the memory above does not "
            "contain enough direct evidence."
            if retrieval_enabled
            else "Answer using only the memory sections above."
        )
    )
    callback = (
        LangChainTokenCallback(token_ledger, "agent")
        if token_ledger is not None
        else None
    )
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if callback is not None:
        config["callbacks"] = [callback]
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user}]},
        config=config,
    )
    messages = result["messages"]
    answer = final_ai_text(messages)
    if token_ledger is not None and callback is not None and callback.recorded_calls == 0:
        token_ledger.record_text("agent", user, answer)
    return answer, collect_tool_trace(messages), len(messages)


def run(args: argparse.Namespace | BeamDeepAgentRunConfig) -> dict[str, Any]:
    if isinstance(args, argparse.Namespace):
        args = BeamDeepAgentRunConfig.from_args(args)

    load_dotenv(args.env_file)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; add it to .env or the environment.")

    use_structured = True
    memory_mode = "chat_deepagent"
    run_id = time.strftime("%Y%m%d-%H%M%S")
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or results_dir / f"memory_agent_{memory_mode}_results_{run_id}.json"
    answers_output_path = args.answers_output or default_answers_output_path(output_path)
    evaluation_output_path = args.evaluation_output or default_evaluation_output_path(
        answers_output_path
    )

    chat = load_json(args.chat)
    probes = select_probes(
        load_json(args.probes),
        question_types=args.question_types,
        max_questions_per_type=args.max_questions_per_type,
    )
    topics = load_json(args.topics)
    topic = load_topic(topics)
    chunks = flatten_chunks(chat)
    message_batches = flatten_message_batches(chat)

    structured_middleware: ChatMemory | None = None
    active_messages: list[AnyMessage] = []
    token_ledger: TokenLedger | None = None

    if use_structured:
        structured_middleware = build_structured_beam_middleware(args)
        token_ledger = structured_middleware.token_ledger or TokenLedger()
        token_ledger.ensure_roles("agent", "judge")

        print(
            "Processing BEAM transcript with public ChatMemory "
            f"({len(message_batches)} user/assistant pair batches)",
            flush=True,
        )
        pending_messages: list[AnyMessage] = []
        pending_chars = 0
        update_threshold_chars = max(4000, args.structured_max_tokens * 4)
        for index, batch in enumerate(message_batches, start=1):
            active_messages.extend(batch)
            pending_messages.extend(batch)
            pending_chars += chat_batch_chars(batch)
            active_messages = active_messages[-args.structured_keep_messages:] if args.structured_keep_messages else []
            should_update = pending_chars >= update_threshold_chars or index == len(message_batches)
            if should_update:
                update_chat_memory(structured_middleware, pending_messages, index)
                pending_messages = []
                pending_chars = 0
                print(
                    f"  structured processed {index}/{len(message_batches)}; "
                    f"active_messages={len(active_messages)}; "
                    f"entries={len(structured_middleware.memory.entries)}",
                    flush=True,
                )

        if args.structured_flush_final and active_messages:
            print(
                "  structured final flush; "
                f"active_messages={len(active_messages)}; "
                f"entries={len(structured_middleware.memory.entries)}",
                flush=True,
            )

    search_tool = make_search_tool(
        memory=structured_middleware,
        default_limit=args.top_k,
        max_hit_chars=args.max_hit_chars,
    )
    assert token_ledger is not None
    agent = build_answer_agent(
        model=args.answer_model,
        search_tool=search_tool,
        system_prompt=build_agent_system_prompt(
            structured_middleware=structured_middleware,
            active_messages=active_messages,
            structured_answer_tokens=args.structured_answer_tokens,
            max_active_context_chars=args.max_active_context_chars,
            retrieval_enabled=True,
        ),
    )
    judge_llm = (
        OpenAIClient(args.judge_model, role="judge", token_ledger=token_ledger)
        if args.judge_model
        else None
    )

    output: dict[str, Any] = {
        "run_id": run_id,
        "source_commit": current_source_commit(),
        "source_state": current_source_state(),
        "config": beam_config_snapshot(args),
        "memory_mode": memory_mode,
        "memory_profile": "chat",
        "chat": str(args.chat),
        "probes": str(args.probes),
        "topics": str(args.topics),
        "output": str(output_path),
        "answers_output": str(answers_output_path),
        "evaluation_output": str(evaluation_output_path) if args.judge_model else None,
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
            len(active_messages) if structured_middleware is not None else None
        ),
        "structured_active_messages": len(active_messages) if structured_middleware is not None else None,
        "results": {},
        "summary": {},
        "token_usage": {},
    }

    total_hits = 0
    total_rubrics = 0
    total_tool_calls = 0
    total_judge_hits = 0
    total_judge_rubrics = 0
    total_judge_score = 0.0
    total_judge_questions = 0
    for question_type, items in probes.items():
        print(f"Answering {question_type}: {len(items)} question(s)", flush=True)
        category_results = []
        category_hits = 0
        category_rubrics = 0
        category_judge_hits = 0
        category_judge_rubrics = 0
        category_judge_score = 0.0
        category_judge_questions = 0

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
                retrieval_enabled=True,
                token_ledger=token_ledger,
            )
            total_tool_calls += len(tool_trace)
            rubric_checks = [rubric_hit(response, line) for line in item.get("rubric", [])]
            hit_count = sum(1 for check in rubric_checks if check["hit"])
            category_hits += hit_count
            category_rubrics += len(rubric_checks)

            judge_checks = []
            judge_hit_count = 0
            question_judge_score = None
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
                question_judge_score = judge_score(judge_checks)
                category_judge_hits += judge_hit_count
                category_judge_rubrics += len(judge_checks)
                if question_judge_score is not None:
                    category_judge_score += question_judge_score
                    category_judge_questions += 1

            judge_score_text = (
                f"{question_judge_score:.3f}" if question_judge_score is not None else "n/a"
            )
            print(
                f"  {question_type}[{item_index}] "
                f"heuristic={hit_count}/{len(rubric_checks)} "
                f"searches={len(tool_trace)}"
                + (
                    f" judge={judge_score_text} ({judge_hit_count}/{len(judge_checks)})"
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
                    "llm_judge_score": round(question_judge_score, 6)
                    if question_judge_score is not None
                    else None,
                    "judge_rubric_hits": judge_hit_count if judge_llm is not None else None,
                    "judge_rubric_total": len(judge_checks) if judge_llm is not None else None,
                }
            )

        total_hits += category_hits
        total_rubrics += category_rubrics
        total_judge_hits += category_judge_hits
        total_judge_rubrics += category_judge_rubrics
        total_judge_score += category_judge_score
        total_judge_questions += category_judge_questions
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
            "judge_score": round(category_judge_score / category_judge_questions, 3)
            if judge_llm is not None and category_judge_questions
            else None,
            "judge_questions": category_judge_questions if judge_llm is not None else None,
        }

    output["token_usage"] = token_ledger.to_dict()

    output["summary"]["overall"] = {
        "chunks_available": len(chunks),
        "chunks_ingested": 0,
        "structured_entries": (
            len(structured_middleware.memory.entries) if structured_middleware is not None else 0
        ),
        "structured_transcript_length": (
            len(active_messages) if structured_middleware is not None else 0
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
        "judge_score": round(total_judge_score / total_judge_questions, 3)
        if judge_llm is not None and total_judge_questions
        else None,
        "judge_questions": total_judge_questions if judge_llm is not None else None,
        "token_usage": token_ledger.to_dict(),
        "note": (
            "Heuristic plus BEAM-style local LLM-as-judge over rubrics."
            if judge_llm is not None
            else "Heuristic only. BEAM-style LLM-as-judge is disabled by --no-judge."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    answers_output_path.parent.mkdir(parents=True, exist_ok=True)
    if judge_llm is not None:
        evaluation_output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with answers_output_path.open("w", encoding="utf-8") as f:
        json.dump(beam_answers_from_results(output["results"]), f, indent=2, ensure_ascii=False)
    if judge_llm is not None:
        with evaluation_output_path.open("w", encoding="utf-8") as f:
            json.dump(
                beam_evaluation_from_results(output["results"]),
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"Wrote results to {output_path}")
    print(f"Wrote BEAM-compatible answers to {answers_output_path}")
    if judge_llm is not None:
        print(f"Wrote BEAM-style evaluation to {evaluation_output_path}")
    print(json.dumps(output["summary"]["overall"], indent=2))
    return output


def parse_args() -> argparse.Namespace:
    config_path, beam_config = beam_config_from_argv()
    product_path, _ = product_config_from_argv()
    defaults = beam_config.to_run_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-config", type=Path, default=config_path)
    parser.add_argument("--product-config", type=Path, default=product_path)
    parser.add_argument("--chat", type=Path, default=defaults["chat"])
    parser.add_argument("--probes", type=Path, default=defaults["probes"])
    parser.add_argument("--topics", type=Path, default=defaults["topics"])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--answers-output",
        type=Path,
        help="Optional BEAM-compatible answers JSON path; defaults next to --output.",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        help="Optional BEAM-style judge evaluation JSON path; used when judge is enabled.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--max-hit-chars", type=int, default=defaults["max_hit_chars"])
    parser.add_argument(
        "--max-active-context-chars",
        type=int,
        default=defaults["max_active_context_chars"],
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=defaults["question_types"],
        help=(
            "BEAM question types to answer. Defaults to contradiction_resolution, "
            "knowledge_update, preference_following, instruction_following, "
            "abstention, and summarization."
        ),
    )
    parser.add_argument(
        "--all-question-types",
        dest="question_types",
        action="store_const",
        const=None,
        help="Run every question type available in the BEAM probe file.",
    )
    parser.add_argument(
        "--max-questions-per-type",
        type=int,
        default=defaults["max_questions_per_type"],
        help="Optional cap per selected question type for quick smoke tests.",
    )
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--answer-model", default=defaults["answer_model"])
    parser.add_argument("--structured-model", default=defaults["structured_model"])
    parser.add_argument(
        "--structured-max-tokens", type=int, default=defaults["structured_max_tokens"]
    )
    parser.add_argument(
        "--structured-max-memory-tokens",
        type=int,
        default=defaults["structured_max_memory_tokens"],
    )
    parser.add_argument(
        "--structured-answer-tokens",
        type=int,
        default=defaults["structured_answer_tokens"],
    )
    parser.add_argument(
        "--structured-evict-fraction",
        type=float,
        default=defaults["structured_evict_fraction"],
    )
    parser.add_argument(
        "--structured-keep-messages",
        type=int,
        default=defaults["structured_keep_messages"],
    )
    parser.add_argument("--no-structured-flush-final", dest="structured_flush_final", action="store_false")
    parser.set_defaults(structured_flush_final=True)
    parser.add_argument("--recursion-limit", type=int, default=defaults["recursion_limit"])
    parser.add_argument(
        "--judge-model",
        default=defaults["judge_model"],
        help="LLM-as-judge model. Defaults to BEAM_JUDGE_MODEL or the answer model default.",
    )
    parser.add_argument(
        "--no-judge",
        dest="judge_model",
        action="store_const",
        const=None,
        help="Disable BEAM-style LLM-as-judge and report only heuristic rubric scores.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

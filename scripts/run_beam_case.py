"""Run one BEAM chat case with structured memory plus local mem0 recall.

This is a smoke-test runner, not the official BEAM evaluator. It uses BEAM's
probing-question rubrics as local ground-truth hints and reports a heuristic
rubric hit rate. Official BEAM scoring uses an LLM-as-judge over the same
rubrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from memory_agent import AGENT_SECTIONS, Mem0LongTermMemory, Memory, MemoryUpdater, OpenAIClient
from memory_agent.langchain_middleware import StructuredMemoryMiddleware


DEFAULT_CHAT_PATH = Path("data/beam/100K/1/chat.json")
DEFAULT_PROBES_PATH = Path("data/beam/100K/1/probing_questions/probing_questions.json")
DEFAULT_TOPICS_PATH = Path("data/beam/topics/100k/100k_topics.json")
DEFAULT_RESULTS_DIR = Path("data/beam/results/100K/1")

STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "and",
    "any",
    "are",
    "based",
    "been",
    "being",
    "but",
    "can",
    "contain",
    "contains",
    "could",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "include",
    "into",
    "its",
    "llm",
    "mention",
    "mentioned",
    "must",
    "not",
    "only",
    "provided",
    "response",
    "should",
    "state",
    "that",
    "the",
    "their",
    "there",
    "this",
    "user",
    "with",
    "would",
    "you",
    "your",
}


@dataclass(frozen=True)
class BeamChunk:
    text: str
    metadata: dict[str, Any]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9_.:-]*", text.lower())
    return {word for word in words if len(word) > 2 and word not in STOPWORDS}


def reference_answer(item: dict[str, Any]) -> str:
    for key in ("ideal_response", "ideal_answer", "answer", "ideal_summary", "expected_compliance"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def rubric_target(rubric_line: str) -> str:
    target = rubric_line.strip()
    target = re.sub(
        r"^LLM response should (?:state|contain|mention):\s*",
        "",
        target,
        flags=re.IGNORECASE,
    )
    return target.strip()


def rubric_hit(response: str, rubric_line: str) -> dict[str, Any]:
    target = rubric_target(rubric_line)
    response_norm = normalize_text(response)
    target_norm = normalize_text(target)
    exact = target_norm in response_norm

    target_words = content_words(target)
    response_words = content_words(response)
    overlap = len(target_words & response_words)
    ratio = overlap / max(1, len(target_words))

    required_numbers = re.findall(r"\d+(?:\.\d+)?", target)
    numbers_present = all(number in response for number in required_numbers)
    hit = exact or ratio >= 0.65 or (numbers_present and ratio >= 0.45)

    return {
        "rubric": rubric_line,
        "target": target,
        "hit": hit,
        "exact": exact,
        "word_overlap_ratio": round(ratio, 3),
    }


def flatten_chunks(chat: list[dict[str, Any]]) -> list[BeamChunk]:
    chunks: list[BeamChunk] = []
    for batch_index, batch in enumerate(chat, start=1):
        batch_number = batch.get("batch_number", batch_index)
        for turn_index, turn in enumerate(batch.get("turns", []), start=1):
            for pair_index in range(0, len(turn), 2):
                pair = turn[pair_index : pair_index + 2]
                if not pair:
                    continue

                ids = [message.get("id") for message in pair if message.get("id") is not None]
                index_values = [
                    message.get("index")
                    for message in pair
                    if isinstance(message.get("index"), str) and message.get("index")
                ]
                lines = [
                    f"BEAM source chat ids: {', '.join(str(chat_id) for chat_id in ids)}",
                    f"Batch: {batch_number}; turn group: {turn_index}; pair: {pair_index // 2 + 1}",
                ]
                if index_values:
                    lines.append(f"Conversation index: {', '.join(index_values)}")

                for message in pair:
                    role = str(message.get("role", "unknown")).upper()
                    content = str(message.get("content", "")).strip()
                    if content:
                        lines.append(f"{role}: {content}")

                chunks.append(
                    BeamChunk(
                        text="\n".join(lines),
                        metadata={
                            "source": "BEAM",
                            "chat_size": "100K",
                            "case_id": "1",
                            "batch_number": batch_number,
                            "turn_group": turn_index,
                            "pair_number": pair_index // 2 + 1,
                            "chat_ids": ",".join(str(chat_id) for chat_id in ids),
                        },
                    )
                )
    return chunks


def flatten_message_batches(chat: list[dict[str, Any]]) -> list[list[AnyMessage]]:
    batches: list[list[AnyMessage]] = []
    for batch_index, batch in enumerate(chat, start=1):
        batch_number = batch.get("batch_number", batch_index)
        for turn_index, turn in enumerate(batch.get("turns", []), start=1):
            for pair_index in range(0, len(turn), 2):
                pair = turn[pair_index : pair_index + 2]
                messages: list[AnyMessage] = []
                for message in pair:
                    role = message.get("role")
                    content = str(message.get("content", "")).strip()
                    if not content:
                        continue
                    message_id = message.get("id")
                    stable_id = (
                        f"beam-{message_id}"
                        if message_id is not None
                        else f"beam-{batch_number}-{turn_index}-{pair_index}-{len(messages)}"
                    )
                    metadata = {
                        "beam_batch_number": batch_number,
                        "beam_turn_group": turn_index,
                        "beam_pair_number": pair_index // 2 + 1,
                        "beam_chat_id": message_id,
                        "beam_index": message.get("index"),
                    }
                    if role == "user":
                        messages.append(
                            HumanMessage(content=content, id=stable_id, additional_kwargs=metadata)
                        )
                    elif role == "assistant":
                        messages.append(
                            AIMessage(content=content, id=stable_id, additional_kwargs=metadata)
                        )
                if messages:
                    batches.append(messages)
    return batches


def load_topic(topics: list[dict[str, Any]], topic_id: int = 1) -> dict[str, Any]:
    for topic in topics:
        if topic.get("id") == topic_id:
            return topic
    return {}


def apply_message_update(messages: list[AnyMessage], update: dict[str, Any] | None) -> list[AnyMessage]:
    if not update:
        return messages

    updated_messages = update.get("messages")
    if not isinstance(updated_messages, list):
        return messages

    if (
        updated_messages
        and isinstance(updated_messages[0], RemoveMessage)
        and updated_messages[0].id == REMOVE_ALL_MESSAGES
    ):
        return list(updated_messages[1:])

    return messages + [m for m in updated_messages if not isinstance(m, RemoveMessage)]


def render_message_tail(messages: list[AnyMessage], max_chars: int) -> str:
    if not messages:
        return "(No active working-context messages.)"

    lines = []
    for message in messages:
        role = "USER" if isinstance(message, HumanMessage) else "ASSISTANT"
        chat_id = message.additional_kwargs.get("beam_chat_id")
        prefix = f"{role}"
        if chat_id is not None:
            prefix += f" chat_id={chat_id}"
        lines.append(f"{prefix}: {message.content}")

    text = "\n\n".join(lines)
    if max_chars > 0 and len(text) > max_chars:
        text = "[truncated]\n" + text[-max_chars:]
    return text


def build_context(hits: list[Any], max_hit_chars: int) -> str:
    if not hits:
        return "No retrieved memory."

    blocks = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit.metadata or {}
        chat_ids = metadata.get("chat_ids", "unknown")
        text = hit.text
        if max_hit_chars > 0 and len(text) > max_hit_chars:
            text = text[:max_hit_chars] + "\n[truncated]"
        blocks.append(f"[Retrieved {index}; chat_ids={chat_ids}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def build_answer_context(
    structured_middleware: StructuredMemoryMiddleware | None,
    active_messages: list[AnyMessage],
    hits: list[Any],
    max_hit_chars: int,
    max_active_context_chars: int,
    structured_answer_tokens: int,
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
        "# Conversation Memory\n"
        "Structured current-state memory generated by StructuredMemoryMiddleware. "
        "Prefer this section if it conflicts with raw long-term recall.\n"
        f"{conversation_memory}\n\n"
        "# Working Conversation Tail\n"
        "Recent active messages not yet evicted into structured memory.\n"
        f"{working_tail}\n\n"
        "# Long-Term Memory\n"
        "Raw mem0 retrieval from the BEAM transcript.\n"
        f"{build_context(hits, max_hit_chars=max_hit_chars)}"
    )


def answer_question(
    llm: OpenAIClient,
    model: str,
    topic: dict[str, Any],
    question_type: str,
    question: str,
    context: str,
) -> str:
    system = (
        "You answer BEAM long-term-memory probing questions. Use only the "
        "provided memory context. The # Conversation Memory section is the "
        "structured current state produced by StructuredMemoryMiddleware; prefer "
        "it if it conflicts with # Long-Term Memory. Use # Long-Term Memory as "
        "raw supporting recall from mem0. If the answer is not supported by the "
        "provided context, say that the provided chat does not contain enough "
        "information. Follow any remembered user instructions that are relevant "
        "to the question."
    )
    topic_text = json.dumps(topic, ensure_ascii=False)
    user = (
        f"Topic metadata:\n{topic_text}\n\n"
        f"Question type: {question_type}\n\n"
        f"Memory context:\n{context}\n\n"
        f"Probing question:\n{question}"
    )
    return llm.complete(system=system, messages=[{"role": "user", "content": user}], model=model)


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(args.env_file)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; add it to .env or the environment.")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    store_dir = args.store_dir or results_dir / f"mem0_store_{run_id}"
    output_path = args.output or results_dir / f"memory_agent_{args.memory_mode}_results_{run_id}.json"

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

    llm = OpenAIClient(args.answer_model)
    output: dict[str, Any] = {
        "run_id": run_id,
        "memory_mode": args.memory_mode,
        "chat": str(args.chat),
        "probes": str(args.probes),
        "topics": str(args.topics),
        "store_dir": str(store_dir),
        "answer_model": args.answer_model,
        "structured_model": args.structured_model if structured_middleware is not None else None,
        "retrieval_top_k": args.top_k,
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
    for question_type, items in probes.items():
        print(f"Answering {question_type}: {len(items)} question(s)", flush=True)
        category_results = []
        category_hits = 0
        category_rubrics = 0

        for item_index, item in enumerate(items):
            question = item["question"]
            hits = memory.search(question, user_id=args.user_id, limit=args.top_k)
            response = answer_question(
                llm=llm,
                model=args.answer_model,
                topic=topic,
                question_type=question_type,
                question=question,
                context=build_answer_context(
                    structured_middleware=structured_middleware,
                    active_messages=active_messages,
                    hits=hits,
                    max_hit_chars=args.max_hit_chars,
                    max_active_context_chars=args.max_active_context_chars,
                    structured_answer_tokens=args.structured_answer_tokens,
                ),
            )
            rubric_checks = [rubric_hit(response, line) for line in item.get("rubric", [])]
            hit_count = sum(1 for check in rubric_checks if check["hit"])
            category_hits += hit_count
            category_rubrics += len(rubric_checks)

            print(
                f"  {question_type}[{item_index}] "
                f"heuristic={hit_count}/{len(rubric_checks)}"
                ,
                flush=True,
            )
            category_results.append(
                {
                    "question": question,
                    "reference_answer": reference_answer(item),
                    "llm_response": response,
                    "retrieved": [
                        {
                            "text": hit.text,
                            "score": hit.score,
                            "metadata": hit.metadata,
                        }
                        for hit in hits
                    ],
                    "structured_memory_used": structured_middleware is not None,
                    "rubric_checks": rubric_checks,
                    "heuristic_rubric_hits": hit_count,
                    "heuristic_rubric_total": len(rubric_checks),
                }
            )

        total_hits += category_hits
        total_rubrics += category_rubrics
        output["results"][question_type] = category_results
        output["summary"][question_type] = {
            "heuristic_rubric_hits": category_hits,
            "heuristic_rubric_total": category_rubrics,
            "heuristic_rubric_rate": round(category_hits / category_rubrics, 3)
            if category_rubrics
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
        "heuristic_rubric_hits": total_hits,
        "heuristic_rubric_total": total_rubrics,
        "heuristic_rubric_rate": round(total_hits / total_rubrics, 3) if total_rubrics else None,
        "note": "Heuristic only. Official BEAM evaluation uses LLM-as-judge over rubrics.",
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
        help="structured_mem0 uses StructuredMemoryMiddleware plus mem0 retrieval; raw_mem0 keeps the old raw mem0-only behavior.",
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
    parser.add_argument("--no-structured-flush-final", dest="structured_flush_final", action="store_false")
    parser.set_defaults(structured_flush_final=True)
    parser.add_argument("--mem0-llm-model", default=os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

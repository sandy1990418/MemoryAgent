"""Run one BEAM chat case with structured memory plus local mem0 recall.

This runner answers one BEAM chat case and writes both a detailed local trace
and BEAM-compatible answer/evaluation files. BEAM's official evaluation scores
each probing-question rubric with an LLM-as-judge on a 0.0/0.5/1.0 scale; the
same rubric-level judge shape is used here by default unless --no-judge is set.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from memory_agent import (
    Mem0LongTermMemory,
    Memory,
    MemoryCompactor,
    MemoryUpdater,
    OpenAIClient,
    TokenLedger,
    get_memory_policy,
)
from memory_agent.models.policy import is_chat_policy
from scripts.beam_models import (
    DEFAULT_RESULTS_DIR,
    BeamChunk,
    BeamRunConfig,
    beam_config_from_argv,
    normalize_beam_profile,
)
from evaluation.beam.chat_case_adapter import BeamChatCaseAdapter
from memory_agent.models.config import ProductMemoryConfig, product_config_from_argv
from memory_agent.structured.middleware import StructuredMemoryMiddleware
from memory_agent.models.sections import sections_for_preset

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

BEAM_TOKEN_ROLES = ("updater", "compactor", "agent", "judge")


def current_source_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() or None


def current_source_state() -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    ).stdout
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    ).stdout
    untracked = sorted(path for path in untracked_output.splitlines() if path)
    digest_input = bytearray(diff)
    for relative_path in untracked:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            continue
        digest_input.extend(f"\nUNTRACKED:{relative_path}\n".encode())
        digest_input.extend(path.read_bytes())
    return {
        "dirty": bool(digest_input),
        "diff_sha256": (
            hashlib.sha256(digest_input).hexdigest() if digest_input else None
        ),
        "untracked": untracked,
    }


def beam_config_snapshot(config: BeamRunConfig) -> dict[str, Any]:
    def jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    return jsonable(asdict(config))


def build_structured_beam_middleware(
    args: argparse.Namespace | BeamRunConfig,
    token_ledger: TokenLedger | None = None,
) -> StructuredMemoryMiddleware:
    """Build one policy-consistent structured-memory stack for BEAM."""
    ledger = token_ledger or TokenLedger()
    memory_policy = get_memory_policy(normalize_beam_profile(args.memory_profile))
    product = ProductMemoryConfig.from_yaml_env(
        getattr(args, "product_config", None) or "configs/product.yaml"
    )
    product_policy = get_memory_policy(product.memory_profile)
    section_preset = (
        product.sections
        if product_policy.name == memory_policy.name
        else memory_policy.section_preset
    )
    sections = sections_for_preset(section_preset)
    updater = MemoryUpdater(
        llm=OpenAIClient(
            args.structured_model,
            role="updater",
            token_ledger=ledger,
        ),
        sections=sections,
        policy=memory_policy,
    )
    compactor = (
        MemoryCompactor(
            llm=OpenAIClient(
                args.structured_model,
                role="compactor",
                token_ledger=ledger,
            ),
            sections=sections,
            policy=memory_policy,
        )
        if is_chat_policy(memory_policy)
        else None
    )
    return StructuredMemoryMiddleware(
        memory=Memory(sections=sections, policy=memory_policy),
        updater=updater,
        policy=memory_policy,
        max_tokens=args.structured_max_tokens,
        evict_fraction=args.structured_evict_fraction,
        keep_messages=args.structured_keep_messages,
        max_memory_tokens=args.structured_max_memory_tokens,
        compactor=compactor,
        compact_min_active_entries=product.compaction_threshold,
    )


# Mirrors BEAM's unified_llm_judge_base_prompt structure.
BEAM_JUDGE_SYSTEM = "You are an expert evaluator."

BEAM_JUDGE_USER_TEMPLATE = """You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): {question}
- RUBRIC CRITERION (what to check): {rubric}
- RESPONSE TO EVALUATE: {response}

## EVALUATION RUBRIC
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be on-topic with respect to the QUESTION and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES
Judge by meaning, not exact wording.
- Accept paraphrases and synonyms that preserve intent.
- Case, punctuation, and whitespace differences must be ignored.
- Numbers, currencies, dates, and durations may appear in equivalent forms.
- If the rubric expects a number or duration, prefer normalized comparison over string matching.

## STYLE NEUTRALITY
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure.
- Do not penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric explicitly mandates it.

## SCORING SCALE
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
- **0.5 (Partial Compliance)**: Partially complies.
- **0.0 (No Compliance)**: Fails to comply.

## EVALUATION INSTRUCTIONS
1. Understand whether the rubric asks for something to be present or absent.
2. Parse compound statements and decide whether all elements are required.
3. Check compliance with the specific rubric criterion.
4. Assign score according to the scoring scale.
5. Provide reasoning for the score.

## OUTPUT FORMAT
Return your evaluation in JSON format with two fields:
{{"score": 1.0, "reason": "explanation of whether the rubric criterion was satisfied and why"}}
NOTE: ONLY output the json object, without any explanation before or after that.
"""

# Mirrors BEAM's answer_generation_for_rag baseline prompt.
BEAM_RAG_ANSWER_SYSTEM = "You are an assistant."

BEAM_RAG_ANSWER_TEMPLATE = """You are an assistant that MUST answer questions using ONLY the information provided in the context below.

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided context.
2. Do NOT invent user history or project facts.
3. Follow durable user instructions found in Conversation Memory.
4. For implementation/how-to requests, you may synthesize code using technologies
   explicitly named in context; distinguish generated guidance from remembered facts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER REQUIREMENTS:
- Be direct and concise.
- For latest/current values, use the latest active memory entry for that subject.
- If a durable instruction requires dependency versions, list only dependencies
  whose versions are present in context; do not output unversioned dependencies.
- When recommendations are constrained by a stored preference (for example,
  lightweight/minimal tools), explicitly state how the answer follows that
  preference and avoids conflicting alternatives.
- For date or duration questions, identify the relevant dated events from context and calculate carefully.
- Abstain only when no relevant memory entry or recent context contains the answer.
- Only output the answer to the question without any explanation.

RESPONSE:
"""


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
    numbers_present = bool(required_numbers) and all(number in response for number in required_numbers)
    hit = exact or ratio >= 0.65 or (numbers_present and ratio >= 0.45)

    return {
        "rubric": rubric_line,
        "target": target,
        "hit": hit,
        "exact": exact,
        "word_overlap_ratio": round(ratio, 3),
    }


def parse_judge_response(response: str) -> dict[str, Any] | None:
    text = response.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence_re = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    for match in fence_re.finditer(text):
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def normalize_judge_score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def normalize_judge_checks(
    parsed: dict[str, Any] | None,
    rubric_lines: list[str],
) -> list[dict[str, Any]]:
    if isinstance(parsed, dict) and isinstance(parsed.get("checks"), list):
        raw_checks = parsed["checks"]
    elif isinstance(parsed, dict) and ("score" in parsed or "passed" in parsed):
        raw_checks = [parsed]
    else:
        raw_checks = None

    if not isinstance(raw_checks, list):
        return [
            {
                "rubric": rubric,
                "target": rubric_target(rubric),
                "score": 0.0,
                "passed": False,
                "reason": "judge response could not be parsed",
            }
            for rubric in rubric_lines
        ]

    checks: list[dict[str, Any]] = []
    for index, rubric in enumerate(rubric_lines):
        raw_check = raw_checks[index] if index < len(raw_checks) else {}
        if not isinstance(raw_check, dict):
            raw_check = {}
        if "score" in raw_check:
            score = normalize_judge_score(raw_check.get("score"))
        else:
            score = 1.0 if bool(raw_check.get("passed")) else 0.0
        checks.append(
            {
                "rubric": rubric,
                "target": rubric_target(rubric),
                "score": score,
                "passed": score >= 1.0,
                "reason": str(raw_check.get("reason") or "").strip(),
            }
        )
    return checks


def judge_response(
    llm: OpenAIClient,
    model: str,
    question_type: str,
    question: str,
    reference: str,
    response: str,
    rubric_lines: list[str],
) -> list[dict[str, Any]]:
    if not rubric_lines:
        return []

    checks: list[dict[str, Any]] = []
    for rubric in rubric_lines:
        prompt = BEAM_JUDGE_USER_TEMPLATE.format(
            question=question,
            rubric=rubric,
            response=response,
        )
        raw = llm.complete(
            system=BEAM_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        checks.extend(normalize_judge_checks(parse_judge_response(raw), [rubric]))
    return checks


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


def flatten_message_batches(chat: list[dict[str, Any]], case_id: str = "1") -> list[list[AnyMessage]]:
    adapter = BeamChatCaseAdapter()
    batches: list[list[AnyMessage]] = []
    for batch_index, batch in enumerate(chat, start=1):
        batch_number = batch.get("batch_number", batch_index)
        for turn_index, turn in enumerate(batch.get("turns", []), start=1):
            for pair_index in range(0, len(turn), 2):
                pair = turn[pair_index : pair_index + 2]
                messages: list[AnyMessage] = []
                for event in adapter.adapt_messages(pair, case_id=case_id):
                    role = event.actor
                    content = event.content.strip()
                    if not content:
                        continue
                    message_id = event.metadata.get("beam_chat_id")
                    stable_id = event.event_id
                    metadata = {
                        "beam_batch_number": batch_number,
                        "beam_turn_group": turn_index,
                        "beam_pair_number": pair_index // 2 + 1,
                        "beam_chat_id": message_id,
                        "beam_index": event.metadata.get("beam_index"),
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


def load_topic(topics: Any, topic_id: int = 1) -> dict[str, Any]:
    if isinstance(topics, dict):
        return topics
    if not isinstance(topics, list):
        return {}
    for topic in topics:
        if isinstance(topic, dict) and topic.get("id") == topic_id:
            return topic
    return {}


def select_probes(
    probes: dict[str, list[dict[str, Any]]],
    question_types: list[str] | tuple[str, ...] | None = None,
    max_questions_per_type: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    selected_keys = set(question_types or probes.keys())
    unknown = sorted(selected_keys - set(probes.keys()))
    if unknown:
        raise ValueError(f"Unknown BEAM question type(s): {', '.join(unknown)}")

    selected: dict[str, list[dict[str, Any]]] = {}
    for question_type, items in probes.items():
        if question_type not in selected_keys:
            continue
        selected[question_type] = (
            list(items[:max_questions_per_type])
            if max_questions_per_type is not None
            else list(items)
        )
    return selected


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
    query: str = "",
    question_type: str = "",
) -> str:
    if structured_middleware is None:
        conversation_memory = "(StructuredMemoryMiddleware was not used.)"
        chronological = "(StructuredMemoryMiddleware was not used.)"
        working_tail = "(No structured working-context tail.)"
    else:
        pinned_by_question_type = {
            "instruction_following": {"preferences"},
            "preference_following": {"preferences"},
            "contradiction_resolution": {"status_changes"},
            "summarization": {"preferences", "goal", "status_changes"},
        }
        budget_by_question_type = {
            "instruction_following": min(structured_answer_tokens, 3500),
            "contradiction_resolution": min(structured_answer_tokens, 3000),
            "summarization": structured_answer_tokens,
        }
        effective_budget = budget_by_question_type.get(
            question_type,
            min(structured_answer_tokens, 2500),
        )
        selected_entries = structured_middleware.memory_selector.select(
            memory=structured_middleware.memory,
            query=query,
            max_tokens=effective_budget,
            include_superseded=question_type == "contradiction_resolution",
            pinned_sections=pinned_by_question_type.get(question_type, frozenset()),
        )
        conversation_memory = (
            structured_middleware.memory.render(
                entries=selected_entries,
                include_superseded=question_type == "contradiction_resolution",
                max_tokens=effective_budget,
            )
            or "(No relevant structured memory entries.)"
        )
        if question_type in {"summarization", "contradiction_resolution"}:
            chronological = structured_middleware.memory.render_chronological(
                entries=None if question_type == "summarization" else selected_entries,
                max_tokens=effective_budget // 2,
                # Identifier entries (versions, dates, paths) drown out the
                # topical mention-order signal ordering questions need.
                exclude_sections={"exact_values"},
            ) or "(No chronological memory entries.)"
        else:
            chronological = "(Chronology omitted for this question type.)"
        working_tail = render_message_tail(active_messages, max_active_context_chars)

    return (
        "# Conversation Memory\n"
        "Structured memory summary.\n"
        f"{conversation_memory}\n\n"
        "# Chronological Order\n"
        "Entries ordered by first mention, earliest first.\n"
        f"{chronological}\n\n"
        "# Working Conversation Tail\n"
        "Recent messages not yet folded into memory.\n"
        f"{working_tail}\n\n"
        "# Long-Term Memory\n"
        "Retrieved raw transcript snippets.\n"
        f"{build_context(hits, max_hit_chars=max_hit_chars)}"
    )


def structured_memory_stats(memory: Memory | None) -> dict[str, Any]:
    if memory is None:
        return {}

    entries = list(memory.entries.values())
    active_entries = [entry for entry in entries if entry.status == "active"]
    section_counts: dict[str, int] = {}
    for entry in active_entries:
        section_counts[entry.section] = section_counts.get(entry.section, 0) + 1

    total_chars = sum(len(entry.text) for entry in active_entries)
    return {
        "total_entries": len(entries),
        "active_entries": len(active_entries),
        "superseded_entries": len(entries) - len(active_entries),
        "section_counts": dict(sorted(section_counts.items())),
        "avg_active_entry_chars": round(total_chars / len(active_entries), 1)
        if active_entries
        else 0,
        "total_active_entry_chars": total_chars,
        "long_active_entries_over_180_chars": sum(
            1 for entry in active_entries if len(entry.text) > 180
        ),
    }


def judge_score(checks: list[dict[str, Any]]) -> float | None:
    if not checks:
        return None
    return sum(normalize_judge_score(check.get("score")) for check in checks) / len(checks)


def target_response_position(response: str, target: str) -> int | None:
    response_lower = response.lower()
    target_lower = target.lower()
    exact_index = response_lower.find(target_lower)
    if exact_index >= 0:
        return exact_index

    target_words = content_words(target)
    if not target_words:
        return None

    positions = [
        response_lower.find(word)
        for word in target_words
        if response_lower.find(word) >= 0
    ]
    if len(positions) / len(target_words) < 0.5:
        return None
    return min(positions)


def event_ordering_score_from_response(
    rubric_lines: list[str],
    response: str,
) -> dict[str, float]:
    targets = [rubric_target(line) for line in rubric_lines]
    positions = [target_response_position(response, target) for target in targets]
    matched = sum(position is not None for position in positions)
    recall = matched / len(targets) if targets else 0.0
    precision = recall
    f1 = recall

    pair_total = len(targets) * (len(targets) - 1) / 2
    pair_score = 0.0
    if pair_total:
        for left_index, left_position in enumerate(positions):
            for right_position in positions[left_index + 1 :]:
                if left_position is None or right_position is None:
                    continue
                if left_position < right_position:
                    pair_score += 1.0
                elif left_position == right_position:
                    pair_score += 0.5
        tau_norm = pair_score / pair_total
    else:
        tau_norm = 1.0 if matched else 0.0

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tau_norm": round(tau_norm, 6),
        "final_score": round(tau_norm * f1, 6),
    }


def beam_answers_from_results(results: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, str]]]:
    return {
        question_type: [
            {
                "question": str(item.get("question", "")),
                "llm_response": str(item.get("llm_response", "")),
            }
            for item in items
        ]
        for question_type, items in results.items()
    }


def beam_evaluation_from_results(
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    evaluation: dict[str, list[dict[str, Any]]] = {}
    for question_type, items in results.items():
        rows = []
        for item in items:
            checks = list(item.get("judge_checks") or [])
            llm_judge_score = judge_score(checks)
            row: dict[str, Any] = {
                "question": item.get("question"),
                "llm_response": item.get("llm_response"),
                "llm_judge_score": round(llm_judge_score, 6)
                if llm_judge_score is not None
                else None,
                "llm_judge_responses": [
                    {
                        "rubric": check.get("rubric"),
                        "score": normalize_judge_score(check.get("score")),
                        "reason": check.get("reason", ""),
                    }
                    for check in checks
                ],
            }
            if question_type == "event_ordering":
                row.update(
                    event_ordering_score_from_response(
                        [str(check.get("rubric", "")) for check in checks],
                        str(item.get("llm_response", "")),
                    )
                )
            rows.append(row)
        evaluation[question_type] = rows
    return evaluation


def default_answers_output_path(output_path: Path) -> Path:
    stem = output_path.stem
    if "_results_" in stem:
        stem = stem.replace("_results_", "_answers_")
    else:
        stem = f"{stem}_answers"
    return output_path.with_name(f"{stem}{output_path.suffix}")


def default_evaluation_output_path(answers_path: Path) -> Path:
    return answers_path.with_name(f"evaluation-{answers_path.name}")


def answer_question(
    llm: OpenAIClient,
    model: str,
    topic: dict[str, Any],
    question_type: str,
    question: str,
    context: str,
) -> str:
    topic_text = json.dumps(topic, ensure_ascii=False)
    prompt_context = (
        f"Topic metadata:\n{topic_text}\n\n"
        f"Question type: {question_type}\n\n"
        f"Memory context:\n{context}"
    )
    user = BEAM_RAG_ANSWER_TEMPLATE.format(
        context=prompt_context,
        question=question,
    )
    return llm.complete(
        system=BEAM_RAG_ANSWER_SYSTEM,
        messages=[{"role": "user", "content": user}],
        model=model,
    )


def run(args: argparse.Namespace | BeamRunConfig) -> dict[str, Any]:
    if isinstance(args, argparse.Namespace):
        args = BeamRunConfig.from_args(args)

    load_dotenv(args.env_file)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; add it to .env or the environment.")

    use_mem0 = args.memory_mode in ("structured_mem0", "raw_mem0")
    use_structured = args.memory_mode in ("structured_mem0", "structured_only")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or results_dir / f"memory_agent_{args.memory_mode}_results_{run_id}.json"
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

    memory = None
    store_dir = None
    if use_mem0:
        store_dir = args.store_dir or results_dir / f"mem0_store_{run_id}"
        os.environ.setdefault("MEM0_DIR", str(store_dir))
        os.environ.setdefault("MEM0_TELEMETRY", "False")

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
    else:
        print("mem0 disabled (structured_only): skipping store and ingestion", flush=True)

    structured_middleware: StructuredMemoryMiddleware | None = None
    active_messages: list[AnyMessage] = []
    token_ledger = TokenLedger()
    token_ledger.ensure_roles(*BEAM_TOKEN_ROLES)

    if use_structured:
        structured_middleware = build_structured_beam_middleware(args, token_ledger)

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

    llm = OpenAIClient(args.answer_model, role="agent", token_ledger=token_ledger)
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
        "memory_mode": args.memory_mode,
        "memory_profile": args.memory_profile,
        "chat": str(args.chat),
        "probes": str(args.probes),
        "topics": str(args.topics),
        "store_dir": str(store_dir) if store_dir is not None else None,
        "output": str(output_path),
        "answers_output": str(answers_output_path),
        "evaluation_output": str(evaluation_output_path) if args.judge_model else None,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
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
        "token_usage": {},
    }

    total_hits = 0
    total_rubrics = 0
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
            hits = (
                memory.search(question, user_id=args.user_id, limit=args.top_k)
                if memory is not None
                else []
            )
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
                    query=question,
                    question_type=question_type,
                ),
            )
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
                f"heuristic={hit_count}/{len(rubric_checks)}"
                + (
                    f" judge={judge_score_text} ({judge_hit_count}/{len(judge_checks)})"
                    if judge_llm is not None
                    else ""
                )
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
        "chunks_ingested": len(chunks) if use_mem0 and not args.skip_ingest else 0,
        "structured_entries": (
            len(structured_middleware.memory.entries) if structured_middleware is not None else 0
        ),
        "structured_transcript_length": (
            len(structured_middleware.transcript) if structured_middleware is not None else 0
        ),
        "structured_active_messages": len(active_messages) if structured_middleware is not None else 0,
        "questions_answered": sum(len(items) for items in probes.values()),
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
    product_path, product_config = product_config_from_argv()
    defaults = beam_config.to_run_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-config", type=Path, default=config_path)
    parser.add_argument("--product-config", type=Path, default=product_path)
    parser.add_argument("--chat", type=Path, default=defaults["chat"])
    parser.add_argument("--probes", type=Path, default=defaults["probes"])
    parser.add_argument("--topics", type=Path, default=defaults["topics"])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--store-dir", type=Path)
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
    parser.add_argument("--user-id", default="beam-100k-case-1")
    parser.add_argument(
        "--memory-mode",
        choices=("structured_only", "structured_mem0", "raw_mem0"),
        default="structured_only",
        help=(
            "structured_only answers from StructuredMemoryMiddleware summaries "
            "without mem0; structured_mem0 adds mem0 retrieval; raw_mem0 skips "
            "structured summaries and uses only mem0 retrieval."
        ),
    )
    parser.add_argument(
        "--memory-profile",
        choices=("chat", "practical", "agent", "eval", "beam"),
        default=product_config.memory_profile,
        help=(
            "Structured memory profile for BEAM ingestion. Defaults to MEMORY_PROFILE "
            "or practical; eval/beam reproduces the legacy broad-section behavior."
        ),
    )
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
    parser.add_argument("--mem0-llm-model", default=defaults["mem0_llm_model"])
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

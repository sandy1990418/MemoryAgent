"""Run a standalone BEAM baseline using the final 10 raw conversation pairs.

The memory payload is a JSON array of ``{"user": ..., "assistant": ...}``
objects in chronological order, with the newest pair last. It is passed to the
existing BEAM answer prompt without summary-memory formatting or retrieval.
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

from memory_agent import OpenAIClient, TokenLedger
from evaluation.beam.regression_report import aggregate_runs
from scripts.beam_models import beam_config_from_argv
from scripts.run_beam_case import (
    BEAM_TOKEN_ROLES,
    answer_question,
    apply_score_ownership,
    beam_answers_from_results,
    beam_evaluation_from_results,
    current_source_commit,
    current_source_state,
    default_answers_output_path,
    default_evaluation_output_path,
    judge_response,
    judge_score,
    load_json,
    reference_answer,
    rubric_hit,
    select_probes,
)

BASELINE_PAIR_COUNT = 10
DEFAULT_RESULTS_ROOT = Path("data/beam/baseline-results/100K")


def discover_case_dirs(
    case_root: Path,
    case_ids: list[str] | None = None,
    start_case: int | None = None,
    end_case: int | None = None,
    max_cases: int | None = None,
) -> list[Path]:
    """Discover valid numeric BEAM case directories in deterministic order."""
    if not case_root.exists():
        raise FileNotFoundError(f"BEAM case root does not exist: {case_root}")

    case_dirs = sorted(
        (path for path in case_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if case_ids:
        wanted = {str(case_id) for case_id in case_ids}
        case_dirs = [path for path in case_dirs if path.name in wanted]
        missing = sorted(wanted - {path.name for path in case_dirs}, key=int)
        if missing:
            raise FileNotFoundError(f"Missing BEAM case id(s): {', '.join(missing)}")
    if start_case is not None:
        case_dirs = [path for path in case_dirs if int(path.name) >= start_case]
    if end_case is not None:
        case_dirs = [path for path in case_dirs if int(path.name) <= end_case]
    if max_cases is not None:
        case_dirs = case_dirs[:max_cases]

    for case_dir in case_dirs:
        required = (
            case_dir / "chat.json",
            case_dir / "topic.json",
            case_dir / "probing_questions" / "probing_questions.json",
        )
        missing_files = [str(path) for path in required if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Case {case_dir.name} is missing: {', '.join(missing_files)}"
            )
    return case_dirs


def build_recent_pair_payload(
    chat: list[dict[str, Any]], pair_count: int = BASELINE_PAIR_COUNT
) -> list[dict[str, str]]:
    """Select complete raw pairs and keep the selected tail oldest-to-newest."""
    if pair_count < 1:
        raise ValueError("pair_count must be at least 1")

    pairs: list[dict[str, str]] = []
    for batch in chat:
        for turn in batch.get("turns", []):
            for index in range(0, len(turn), 2):
                raw_pair = turn[index : index + 2]
                if len(raw_pair) != 2:
                    continue
                user, assistant = raw_pair
                if user.get("role") != "user" or assistant.get("role") != "assistant":
                    continue
                pairs.append(
                    {
                        "user": str(user.get("content", "")),
                        "assistant": str(assistant.get("content", "")),
                    }
                )
    return pairs[-pair_count:]


def _config_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(args.env_file)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; add it to .env or the environment.")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or (
        args.results_dir / f"recent_pairs_baseline_results_{run_id}.json"
    )
    answers_path = args.answers_output or default_answers_output_path(output_path)
    evaluation_path = args.evaluation_output or default_evaluation_output_path(
        answers_path
    )

    chat = load_json(args.chat)
    probes = select_probes(
        load_json(args.probes),
        question_types=args.question_types,
        max_questions_per_type=args.max_questions_per_type,
    )
    payload = build_recent_pair_payload(chat, args.pair_count)
    # This exact string is supplied as the answer context. No headings, summary,
    # selector output, retrieval result, or working-tail text is added here.
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)

    token_ledger = TokenLedger()
    token_ledger.ensure_roles(*BEAM_TOKEN_ROLES)
    answer_llm = OpenAIClient(
        args.answer_model, role="agent", token_ledger=token_ledger
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
        "config": _config_snapshot(args),
        "memory_mode": "recent_pairs_baseline",
        "chat": str(args.chat),
        "probes": str(args.probes),
        "topics": str(args.topics),
        "output": str(output_path),
        "answers_output": str(answers_path),
        "evaluation_output": str(evaluation_path) if judge_llm else None,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "baseline_payload": payload,
        "baseline_pair_count": len(payload),
        "baseline_payload_chars": len(payload_json),
        # Compatibility fields keep aggregate comparison schema stable.
        "structured_memory": None,
        "structured_memory_entries": [],
        "compactor_metrics": None,
        "token_usage": {},
        "results": {},
        "summary": {},
    }

    total_hits = total_rubrics = 0
    total_judge_hits = total_judge_rubrics = total_judge_questions = 0
    total_judge_score = 0.0

    for question_type, items in probes.items():
        category_results: list[dict[str, Any]] = []
        category_hits = category_rubrics = 0
        category_judge_hits = category_judge_rubrics = category_judge_questions = 0
        category_judge_score = 0.0
        print(f"Answering {question_type}: {len(items)} question(s)", flush=True)

        for item_index, item in enumerate(items):
            question = str(item["question"])
            answer_started = time.perf_counter()
            response = answer_question(
                llm=answer_llm,
                model=args.answer_model,
                question=question,
                context=payload_json,
            )
            answer_elapsed = round(time.perf_counter() - answer_started, 6)
            rubric_checks = [rubric_hit(response, line) for line in item.get("rubric", [])]
            hit_count = sum(bool(check["hit"]) for check in rubric_checks)
            category_hits += hit_count
            category_rubrics += len(rubric_checks)

            judge_checks = None
            question_judge_score = None
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
                judge_hit_count = sum(bool(check["passed"]) for check in judge_checks)
                question_judge_score = judge_score(judge_checks)
                category_judge_hits += judge_hit_count
                category_judge_rubrics += len(judge_checks)
                if question_judge_score is not None:
                    category_judge_score += question_judge_score
                    category_judge_questions += 1

            category_results.append(
                {
                    "question": question,
                    "reference_answer": reference_answer(item),
                    "llm_response": response,
                    "retrieved": [],
                    "structured_memory_used": False,
                    "selected_memory_ids": None,
                    "answer_context": payload_json,
                    "answer_elapsed_seconds": answer_elapsed,
                    "rubric_checks": rubric_checks,
                    "heuristic_rubric_hits": hit_count,
                    "heuristic_rubric_total": len(rubric_checks),
                    "judge_checks": judge_checks,
                    "llm_judge_score": round(question_judge_score, 6)
                    if question_judge_score is not None
                    else None,
                    "judge_rubric_hits": judge_hit_count if judge_llm else None,
                    "judge_rubric_total": len(judge_checks) if judge_checks is not None else None,
                }
            )
            print(
                f"  {question_type}[{item_index}] heuristic="
                f"{hit_count}/{len(rubric_checks)}"
                + (
                    f" judge={question_judge_score:.3f}"
                    if question_judge_score is not None
                    else ""
                ),
                flush=True,
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
            "judge_rubric_hits": category_judge_hits if judge_llm else None,
            "judge_rubric_total": category_judge_rubrics if judge_llm else None,
            "judge_rubric_rate": round(category_judge_hits / category_judge_rubrics, 3)
            if judge_llm and category_judge_rubrics
            else None,
            "judge_score": round(category_judge_score / category_judge_questions, 3)
            if judge_llm and category_judge_questions
            else None,
            "judge_questions": category_judge_questions if judge_llm else None,
        }

    output["token_usage"] = token_ledger.to_dict()
    output["summary"]["overall"] = {
        "chunks_available": 0,
        "chunks_ingested": 0,
        "structured_entries": 0,
        "structured_transcript_length": 0,
        "structured_active_messages": 0,
        "structured_elapsed_seconds": 0.0,
        "questions_answered": sum(len(items) for items in probes.values()),
        "structured_memory_stats": {},
        "baseline_pairs": len(payload),
        "baseline_payload_chars": len(payload_json),
        "heuristic_rubric_hits": total_hits,
        "heuristic_rubric_total": total_rubrics,
        "heuristic_rubric_rate": round(total_hits / total_rubrics, 3)
        if total_rubrics
        else None,
        "judge_rubric_hits": total_judge_hits if judge_llm else None,
        "judge_rubric_total": total_judge_rubrics if judge_llm else None,
        "judge_rubric_rate": round(total_judge_hits / total_judge_rubrics, 3)
        if judge_llm and total_judge_rubrics
        else None,
        "judge_score": round(total_judge_score / total_judge_questions, 3)
        if judge_llm and total_judge_questions
        else None,
        "judge_questions": total_judge_questions if judge_llm else None,
        "token_usage": token_ledger.to_dict(),
        "note": "Standalone final-10-pair raw JSON baseline.",
    }
    apply_score_ownership(output, "production")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
    with answers_path.open("w", encoding="utf-8") as file:
        json.dump(
            beam_answers_from_results(output["results"]),
            file,
            indent=2,
            ensure_ascii=False,
        )
    if judge_llm is not None:
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        with evaluation_path.open("w", encoding="utf-8") as file:
            json.dump(
                beam_evaluation_from_results(output["results"]),
                file,
                indent=2,
                ensure_ascii=False,
            )
    print(f"Wrote baseline results to {output_path}")
    return output


def _case_args(
    args: argparse.Namespace,
    case_dir: Path,
    batch_run_id: str,
    repeat: int,
) -> argparse.Namespace:
    case_results_dir = args.results_root / case_dir.name
    output = case_results_dir / (
        f"{batch_run_id}_repeat-{repeat}_recent-pairs-baseline.json"
    )
    values = vars(args) | {
        "chat": case_dir / "chat.json",
        "probes": case_dir / "probing_questions" / "probing_questions.json",
        "topics": case_dir / "topic.json",
        "results_dir": case_results_dir,
        "output": output,
        "answers_output": None,
        "evaluation_output": None,
    }
    return argparse.Namespace(**values)


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Run every selected case and write an aggregate comparison manifest."""
    case_dirs = discover_case_dirs(
        args.case_root,
        case_ids=args.case_ids,
        start_case=args.start_case,
        end_case=args.end_case,
        max_cases=args.max_cases,
    )
    if not case_dirs:
        raise RuntimeError("No BEAM cases selected.")
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")

    args.results_root.mkdir(parents=True, exist_ok=True)
    batch_run_id = time.strftime("%Y%m%d-%H%M%S")
    manifest_path = args.results_root / f"batch_manifest_{batch_run_id}.json"
    manifest: dict[str, Any] = {
        "run_id": batch_run_id,
        "runner": "recent_pairs_baseline",
        "case_root": str(args.case_root),
        "results_root": str(args.results_root),
        "case_ids": [path.name for path in case_dirs],
        "repeats": args.repeats,
        "pair_count": args.pair_count,
        "question_types": args.question_types,
        "max_questions_per_type": args.max_questions_per_type,
        "config": _config_snapshot(args),
        "cases": [],
    }
    detailed_results: list[dict[str, Any]] = []
    total = len(case_dirs) * args.repeats
    completed = 0

    for repeat in range(1, args.repeats + 1):
        for case_dir in case_dirs:
            completed += 1
            print(
                f"\n=== Baseline case {case_dir.name} repeat "
                f"{repeat}/{args.repeats} ({completed}/{total}) ===",
                flush=True,
            )
            started = time.perf_counter()
            try:
                result = run(_case_args(args, case_dir, batch_run_id, repeat))
                detailed_results.append(result)
                manifest["cases"].append(
                    {
                        "case_id": case_dir.name,
                        "repeat": repeat,
                        "status": "ok",
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "output": result["output"],
                        "answers_output": result["answers_output"],
                        "evaluation_output": result["evaluation_output"],
                        "summary": result["summary"]["overall"],
                        "token_usage": result["token_usage"],
                    }
                )
            except Exception as exc:
                manifest["cases"].append(
                    {
                        "case_id": case_dir.name,
                        "repeat": repeat,
                        "status": "error",
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "error": str(exc),
                    }
                )
                print(f"Case {case_dir.name} failed: {exc}", flush=True)

            manifest["aggregate"] = aggregate_runs(detailed_results)
            manifest["manifest_path"] = str(manifest_path)
            with manifest_path.open("w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, ensure_ascii=False)

            if args.stop_on_error and manifest["cases"][-1]["status"] == "error":
                return manifest

    print(f"Wrote batch manifest to {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    config_path, config = beam_config_from_argv()
    defaults = config.to_run_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-config", type=Path, default=config_path)
    parser.add_argument("--case-root", type=Path, default=config.data_path.parent)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--start-case", type=int)
    parser.add_argument("--end-case", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--pair-count", type=int, default=BASELINE_PAIR_COUNT)
    parser.add_argument("--question-types", nargs="+", default=defaults["question_types"])
    parser.add_argument(
        "--all-question-types",
        dest="question_types",
        action="store_const",
        const=None,
    )
    parser.add_argument(
        "--max-questions-per-type",
        type=int,
        default=defaults["max_questions_per_type"],
    )
    parser.add_argument("--answer-model", default=defaults["answer_model"])
    parser.add_argument("--judge-model", default=defaults["judge_model"])
    parser.add_argument("--no-judge", dest="judge_model", action="store_const", const=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_batch(parse_args())

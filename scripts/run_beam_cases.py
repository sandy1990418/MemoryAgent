from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory_agent.models.beam import (
    DEFAULT_BEAM_JUDGE_MODEL,
    DEFAULT_BEAM_MEMORY_MODEL,
    DEFAULT_BEAM_MODEL,
    DEFAULT_MEM0_LLM_MODEL,
    BeamDeepAgentRunConfig,
    BeamRunConfig,
)
from scripts.run_beam_case import run as run_standard_case
from scripts.run_beam_case_deepagent import run as run_deepagent_case


DEFAULT_CASE_ROOT = Path("BEAM/chats/100K")
DEFAULT_RESULTS_ROOT = Path("data/beam/results/100K")


def discover_case_dirs(
    case_root: Path,
    case_ids: list[str] | None = None,
    start_case: int | None = None,
    end_case: int | None = None,
    max_cases: int | None = None,
) -> list[Path]:
    if not case_root.exists():
        raise FileNotFoundError(f"BEAM case root does not exist: {case_root}")

    dirs = sorted(
        [path for path in case_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    if case_ids:
        wanted = {str(case_id) for case_id in case_ids}
        dirs = [path for path in dirs if path.name in wanted]
        missing = sorted(wanted - {path.name for path in dirs}, key=int)
        if missing:
            raise FileNotFoundError(f"Missing BEAM case id(s): {', '.join(missing)}")

    if start_case is not None:
        dirs = [path for path in dirs if int(path.name) >= start_case]
    if end_case is not None:
        dirs = [path for path in dirs if int(path.name) <= end_case]
    if max_cases is not None:
        dirs = dirs[:max_cases]

    for path in dirs:
        required = [
            path / "chat.json",
            path / "topic.json",
            path / "probing_questions" / "probing_questions.json",
        ]
        missing_files = [str(file) for file in required if not file.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Case {path.name} is missing required file(s): {', '.join(missing_files)}"
            )

    return dirs


def case_config(args: argparse.Namespace, case_dir: Path) -> BeamRunConfig:
    case_id = case_dir.name
    results_dir = args.results_root / case_id
    store_dir = args.store_root / case_id if args.store_root else results_dir / "mem0_store"
    common: dict[str, Any] = {
        "chat": case_dir / "chat.json",
        "probes": case_dir / "probing_questions" / "probing_questions.json",
        "topics": case_dir / "topic.json",
        "results_dir": results_dir,
        "store_dir": store_dir,
        "env_file": args.env_file,
        "user_id": f"beam-100k-case-{case_id}",
        "memory_mode": args.memory_mode or "structured_only",
        "top_k": args.top_k,
        "max_hit_chars": args.max_hit_chars,
        "max_active_context_chars": args.max_active_context_chars,
        "skip_ingest": args.skip_ingest,
        "answer_model": args.answer_model,
        "structured_model": args.structured_model,
        "structured_max_tokens": args.structured_max_tokens,
        "structured_max_memory_tokens": args.structured_max_memory_tokens,
        "structured_answer_tokens": args.structured_answer_tokens,
        "structured_evict_fraction": args.structured_evict_fraction,
        "structured_keep_messages": args.structured_keep_messages,
        "structured_flush_final": args.structured_flush_final,
        "mem0_llm_model": args.mem0_llm_model,
        "judge_model": args.judge_model,
        "question_types": args.question_types,
        "max_questions_per_type": args.max_questions_per_type,
    }
    if args.runner == "deepagent":
        return BeamDeepAgentRunConfig(**common, recursion_limit=args.recursion_limit)
    return BeamRunConfig(**common)


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    case_dirs = discover_case_dirs(
        case_root=args.case_root,
        case_ids=args.case_ids,
        start_case=args.start_case,
        end_case=args.end_case,
        max_cases=args.max_cases,
    )
    if not case_dirs:
        raise RuntimeError("No BEAM cases selected.")

    args.results_root.mkdir(parents=True, exist_ok=True)
    if args.store_root:
        args.store_root.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "runner": args.runner,
        "case_root": str(args.case_root),
        "results_root": str(args.results_root),
        "case_ids": [path.name for path in case_dirs],
        "question_types": args.question_types,
        "max_questions_per_type": args.max_questions_per_type,
        "cases": [],
    }

    runner = run_deepagent_case if args.runner == "deepagent" else run_standard_case
    total = len(case_dirs)
    for index, case_dir in enumerate(case_dirs, start=1):
        case_id = case_dir.name
        print(f"\n=== BEAM case {case_id} ({index}/{total}) ===", flush=True)
        started = time.time()
        try:
            result = runner(case_config(args, case_dir))
            overall = result.get("summary", {}).get("overall", {})
            manifest["cases"].append(
                {
                    "case_id": case_id,
                    "status": "ok",
                    "elapsed_seconds": round(time.time() - started, 2),
                    "results_dir": str(args.results_root / case_id),
                    "output": result.get("output"),
                    "answers_output": result.get("answers_output"),
                    "evaluation_output": result.get("evaluation_output"),
                    "summary": overall,
                }
            )
        except Exception as exc:
            manifest["cases"].append(
                {
                    "case_id": case_id,
                    "status": "error",
                    "elapsed_seconds": round(time.time() - started, 2),
                    "error": str(exc),
                }
            )
            print(f"Case {case_id} failed: {exc}", flush=True)
            if args.stop_on_error:
                break

        manifest_path = args.results_root / f"batch_manifest_{run_id}.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    manifest["manifest_path"] = str(args.results_root / f"batch_manifest_{run_id}.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--start-case", type=int)
    parser.add_argument("--end-case", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--runner", choices=("standard", "deepagent"), default="standard")
    parser.add_argument(
        "--memory-mode",
        choices=("structured_only", "structured_mem0", "raw_mem0"),
        help=(
            "Defaults to structured_only, which uses only summary memory and no mem0."
        ),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-hit-chars", type=int, default=6000)
    parser.add_argument("--max-active-context-chars", type=int, default=12000)
    parser.add_argument(
        "--question-types",
        nargs="+",
        help="Optional BEAM question types to answer.",
    )
    parser.add_argument(
        "--max-questions-per-type",
        type=int,
        help="Optional cap per selected question type for quick smoke tests.",
    )
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--answer-model", default=DEFAULT_BEAM_MODEL)
    parser.add_argument("--structured-model", default=DEFAULT_BEAM_MEMORY_MODEL)
    parser.add_argument("--structured-max-tokens", type=int, default=12000)
    parser.add_argument("--structured-max-memory-tokens", type=int, default=3000)
    parser.add_argument("--structured-answer-tokens", type=int, default=4000)
    parser.add_argument("--structured-evict-fraction", type=float, default=0.5)
    parser.add_argument("--structured-keep-messages", type=int, default=2)
    parser.add_argument("--no-structured-flush-final", dest="structured_flush_final", action="store_false")
    parser.set_defaults(structured_flush_final=True)
    parser.add_argument("--mem0-llm-model", default=DEFAULT_MEM0_LLM_MODEL)
    parser.add_argument("--recursion-limit", type=int, default=50)
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_BEAM_JUDGE_MODEL,
        help="LLM-as-judge model. Defaults to BEAM_JUDGE_MODEL or the answer model default.",
    )
    parser.add_argument(
        "--no-judge",
        dest="judge_model",
        action="store_const",
        const=None,
        help="Disable BEAM-style LLM-as-judge and report only heuristic rubric scores.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    summary = run_batch(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))

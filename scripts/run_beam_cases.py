from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.beam_models import (
    BeamDeepAgentRunConfig,
    BeamRunConfig,
    beam_config_from_argv,
)
from memory_agent.models.config import product_config_from_argv
from scripts.run_beam_case import run as run_standard_case
from scripts.run_beam_case_deepagent import run as run_deepagent_case
from evaluation.beam.regression_report import aggregate_runs, compare_aggregates


DEFAULT_CASE_ROOT = Path("BEAM/chats/100K")
DEFAULT_RESULTS_ROOT = Path("data/beam/results/100K")
DEFAULT_SPLIT_FILE = Path("evaluation/beam/splits-100k-v1.json")


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
        "beam_config": getattr(args, "beam_config", None),
        "product_config": getattr(args, "product_config", None),
        "chat": case_dir / "chat.json",
        "probes": case_dir / "probing_questions" / "probing_questions.json",
        "topics": case_dir / "topic.json",
        "results_dir": results_dir,
        "store_dir": store_dir,
        "env_file": args.env_file,
        "user_id": f"beam-100k-case-{case_id}",
        "memory_mode": args.memory_mode or "structured_only",
        "memory_profile": args.memory_profile,
        "top_k": args.top_k,
        "max_hit_chars": args.max_hit_chars,
        "max_active_context_chars": args.max_active_context_chars,
        "skip_ingest": args.skip_ingest,
        "routing_mode": args.routing_mode,
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


def replay_snapshot_lookup(baseline_manifest: dict[str, Any]) -> dict[tuple[str, int], str]:
    """Map (case_id, repeat) to the baseline run's frozen memory snapshot."""
    lookup: dict[tuple[str, int], str] = {}
    for case in baseline_manifest.get("cases", []):
        snapshot = case.get("memory_snapshot")
        if case.get("status") == "ok" and snapshot:
            lookup[(str(case["case_id"]), int(case["repeat"]))] = snapshot
    return lookup


def resolve_replay_snapshot(
    lookup: dict[tuple[str, int], str], case_id: str, repeat: int
) -> str:
    exact = lookup.get((case_id, repeat))
    if exact is not None:
        return exact
    case_repeats = sorted(
        (key_repeat, path) for (key_case, key_repeat), path in lookup.items()
        if key_case == case_id
    )
    if case_repeats:
        return case_repeats[0][1]
    raise FileNotFoundError(
        f"replay manifest has no memory snapshot for case {case_id}"
    )


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    replay_lookup: dict[tuple[str, int], str] | None = None
    if args.replay_manifest:
        if args.runner != "standard":
            raise ValueError("--replay-manifest is only supported by the standard runner")
        replay_lookup = replay_snapshot_lookup(
            json.loads(args.replay_manifest.read_text(encoding="utf-8"))
        )
    split_definition = None
    if args.split:
        split_definition = json.loads(args.split_file.read_text(encoding="utf-8"))
        split_ids = [str(value) for value in split_definition[args.split]]
        if args.case_ids and set(args.case_ids) != set(split_ids):
            raise ValueError("--case-ids cannot override the frozen --split membership")
        args.case_ids = split_ids
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
        "split": args.split,
        "split_file": str(args.split_file) if args.split else None,
        "split_definition": split_definition,
        "repeats": args.repeats,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cases": [],
    }

    runner = run_deepagent_case if args.runner == "deepagent" else run_standard_case
    total = len(case_dirs) * args.repeats
    completed = 0
    detailed_results: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
      for case_dir in case_dirs:
        completed += 1
        case_id = case_dir.name
        print(
            f"\n=== BEAM case {case_id} repeat {repeat}/{args.repeats} "
            f"({completed}/{total}) ===",
            flush=True,
        )
        started = time.time()
        try:
            config = case_config(args, case_dir)
            output = args.results_root / case_id / f"{run_id}_repeat-{repeat}_{args.routing_mode}.json"
            overrides: dict[str, Any] = {"output": output}
            if replay_lookup is not None:
                overrides["replay_memory"] = Path(
                    resolve_replay_snapshot(replay_lookup, case_id, repeat)
                )
            result = runner(replace(config, **overrides))
            detailed_results.append(result)
            overall = result.get("summary", {}).get("overall", {})
            manifest["cases"].append(
                {
                    "case_id": case_id,
                    "repeat": repeat,
                    "status": "ok",
                    "elapsed_seconds": round(time.time() - started, 2),
                    "results_dir": str(args.results_root / case_id),
                    "output": result.get("output"),
                    "memory_snapshot": result.get("memory_snapshot_output"),
                    "replay_memory": result.get("replay_memory"),
                    "answers_output": result.get("answers_output"),
                    "evaluation_output": result.get("evaluation_output"),
                    "summary": overall,
                    "source_commit": result.get("source_commit"),
                    "source_state": result.get("source_state"),
                    "config": result.get("config"),
                }
            )
        except Exception as exc:
            manifest["cases"].append(
                {
                    "case_id": case_id,
                    "repeat": repeat,
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

      if args.stop_on_error and manifest["cases"] and manifest["cases"][-1]["status"] == "error":
          break

    manifest["aggregate"] = aggregate_runs(detailed_results)
    if args.baseline_manifest:
        baseline = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
        manifest["comparison"] = compare_aggregates(
            baseline["aggregate"], manifest["aggregate"]
        )
        manifest["baseline_manifest"] = str(args.baseline_manifest)

    manifest["manifest_path"] = str(args.results_root / f"batch_manifest_{run_id}.json")
    with Path(manifest["manifest_path"]).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def parse_args() -> argparse.Namespace:
    config_path, beam_config = beam_config_from_argv()
    product_path, product_config = product_config_from_argv()
    defaults = beam_config.to_run_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-config", type=Path, default=config_path)
    parser.add_argument("--product-config", type=Path, default=product_path)
    parser.add_argument("--case-root", type=Path, default=beam_config.data_path.parent)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--start-case", type=int)
    parser.add_argument("--end-case", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument(
        "--split", choices=("development", "validation", "holdout"),
        help="Use immutable case membership from --split-file.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        help=(
            "Batch manifest of a prior run whose frozen memory snapshots are "
            "replayed instead of re-ingesting, for paired A/B comparisons."
        ),
    )
    parser.add_argument("--runner", choices=("standard", "deepagent"), default="standard")
    parser.add_argument(
        "--memory-mode",
        choices=("structured_only", "structured_mem0", "raw_mem0"),
        help=(
            "Defaults to structured_only, which uses only summary memory and no mem0."
        ),
    )
    parser.add_argument(
        "--memory-profile",
        choices=("chat", "practical", "agent", "eval", "beam"),
        default=product_config.memory_profile,
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
        help="Run every question type available in each BEAM probe file.",
    )
    parser.add_argument(
        "--max-questions-per-type",
        type=int,
        default=defaults["max_questions_per_type"],
        help="Optional cap per selected question type for quick smoke tests.",
    )
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument(
        "--routing-mode", choices=("production", "oracle"), default="production"
    )
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
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    summary = run_batch(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))

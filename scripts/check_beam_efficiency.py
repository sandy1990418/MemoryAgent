"""Fail-fast evaluator for quality-preserving BEAM token efficiency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(
    result: dict,
    *,
    min_judge_score: float,
    max_active_entries: int,
    max_agent_tokens: int,
    max_updater_tokens: int,
    max_memory_chars: int | None = None,
    max_avg_entry_chars: float | None = None,
    max_long_entries: int | None = None,
) -> list[str]:
    raw_summary = result["summary"]
    summary = raw_summary.get("overall", raw_summary)
    usage = result["token_usage"]
    checks = {
        "judge_score": (float(summary["judge_score"]), ">=", min_judge_score),
        "active_entries": (
            int(summary["structured_memory_stats"]["active_entries"]),
            "<=",
            max_active_entries,
        ),
        "agent_tokens": (int(usage["agent"]["total_tokens"]), "<=", max_agent_tokens),
        "updater_tokens": (
            int(usage["updater"]["total_tokens"]),
            "<=",
            max_updater_tokens,
        ),
    }
    memory_stats = summary["structured_memory_stats"]
    optional_checks = {
        "memory_chars": (memory_stats.get("total_active_entry_chars"), "<=", max_memory_chars),
        "avg_entry_chars": (memory_stats.get("avg_active_entry_chars"), "<=", max_avg_entry_chars),
        "long_entries": (memory_stats.get("long_active_entries_over_180_chars"), "<=", max_long_entries),
    }
    checks.update({
        name: values
        for name, values in optional_checks.items()
        if values[0] is not None and values[2] is not None
    })
    failures = []
    for name, (actual, operator, expected) in checks.items():
        passed = actual >= expected if operator == ">=" else actual <= expected
        print(f"{'PASS' if passed else 'FAIL'} {name}: {actual} {operator} {expected}")
        if not passed:
            failures.append(name)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--min-judge-score", type=float, default=0.58)
    parser.add_argument("--max-active-entries", type=int, default=40)
    parser.add_argument("--max-agent-tokens", type=int, default=71000)
    parser.add_argument("--max-updater-tokens", type=int, default=180000)
    parser.add_argument("--max-memory-chars", type=int)
    parser.add_argument("--max-avg-entry-chars", type=float)
    parser.add_argument("--max-long-entries", type=int)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    failures = evaluate(
        result,
        min_judge_score=args.min_judge_score,
        max_active_entries=args.max_active_entries,
        max_agent_tokens=args.max_agent_tokens,
        max_updater_tokens=args.max_updater_tokens,
        max_memory_chars=args.max_memory_chars,
        max_avg_entry_chars=args.max_avg_entry_chars,
        max_long_entries=args.max_long_entries,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

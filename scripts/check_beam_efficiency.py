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
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    failures = evaluate(
        result,
        min_judge_score=args.min_judge_score,
        max_active_entries=args.max_active_entries,
        max_agent_tokens=args.max_agent_tokens,
        max_updater_tokens=args.max_updater_tokens,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

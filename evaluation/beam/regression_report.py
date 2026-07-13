"""Cross-case BEAM diagnostics kept strictly in the evaluation layer."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any, Iterable


FAILURE_STAGES = (
    "ingestion_update_failure",
    "compaction_failure",
    "selection_failure",
    "answer_generation_failure",
    "judge_evaluation_failure",
    "benchmark_specific_artifact",
    "unattributed",
)


def _words(text: str) -> set[str]:
    import re

    return {
        word
        for word in re.findall(r"[a-z0-9_.:/-]+", text.lower())
        if len(word) > 2
    }


def _evidence_overlap(target: str, text: str) -> float:
    expected = _words(target)
    return len(expected & _words(text)) / max(1, len(expected))


def attribute_question_failure(
    item: dict[str, Any], *, full_memory: str, compactor_metrics: dict[str, Any] | None
) -> dict[str, Any]:
    """Attribute a failed question from persisted stage evidence.

    Rubrics are consumed only here, in evaluation code. Production components
    never receive them.
    """

    score = item.get("llm_judge_score")
    if score is not None and float(score) >= 1.0:
        return {"stage": "passed", "confidence": 1.0, "evidence": []}

    targets = [str(check.get("target") or "") for check in item.get("rubric_checks", [])]
    context = str(item.get("answer_context") or "")
    response = str(item.get("llm_response") or "")
    memory_overlap = max((_evidence_overlap(target, full_memory) for target in targets), default=0.0)
    context_overlap = max((_evidence_overlap(target, context) for target in targets), default=0.0)
    response_overlap = max((_evidence_overlap(target, response) for target in targets), default=0.0)
    heuristic_passed = any(bool(check.get("hit")) for check in item.get("rubric_checks", []))
    judge_score = float(score) if score is not None else None
    evidence = {
        "memory_overlap": round(memory_overlap, 3),
        "context_overlap": round(context_overlap, 3),
        "response_overlap": round(response_overlap, 3),
        "selected_memory_ids": item.get("selected_memory_ids"),
    }

    # A weak lexical heuristic missing a paraphrase is not a judge failure.
    # Flag evaluation disagreement only when the deterministic check found the
    # target in the response but the LLM judge assigned zero credit.
    if heuristic_passed and judge_score == 0.0 and response_overlap >= 0.65:
        return {"stage": "judge_evaluation_failure", "confidence": 0.7, "evidence": evidence}
    if memory_overlap < 0.35:
        return {"stage": "ingestion_update_failure", "confidence": 0.7, "evidence": evidence}
    if context_overlap < 0.35:
        return {"stage": "selection_failure", "confidence": 0.75, "evidence": evidence}
    if response_overlap < 0.35 or (score is not None and float(score) < 1.0):
        return {"stage": "answer_generation_failure", "confidence": 0.7, "evidence": evidence}
    if compactor_metrics and (
        compactor_metrics.get("failed_compactions", 0)
        or compactor_metrics.get("rejected_compactions", 0)
    ):
        return {"stage": "compaction_failure", "confidence": 0.5, "evidence": evidence}
    return {"stage": "unattributed", "confidence": 0.25, "evidence": evidence}


def aggregate_runs(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    runs = list(results)
    by_type: dict[str, list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    token_totals: Counter[str] = Counter()
    active_entries: list[int] = []
    memory_chars: list[int] = []
    elapsed: list[float] = []
    compactor_calls = 0
    updater_calls = 0
    selected_counts: list[int] = []

    for result in runs:
        overall = result.get("summary", {}).get("overall", {})
        stats = overall.get("structured_memory_stats", {})
        active_entries.append(int(stats.get("active_entries", 0)))
        memory_chars.append(int(stats.get("total_active_entry_chars", 0)))
        elapsed.append(float(overall.get("structured_elapsed_seconds", 0.0)))
        usage = result.get("token_usage", {})
        for role, values in usage.items():
            token_totals[f"{role}_tokens"] += int(values.get("total_tokens", 0))
        compactor_calls += int(usage.get("compactor", {}).get("calls", 0))
        updater_calls += int(usage.get("updater", {}).get("calls", 0))
        full_memory = str(result.get("structured_memory") or "")
        compactor_metrics = result.get("compactor_metrics")
        for question_type, items in result.get("results", {}).items():
            for index, item in enumerate(items):
                score = item.get("llm_judge_score")
                selected = item.get("selected_memory_ids")
                if isinstance(selected, list):
                    selected_counts.append(len(selected))
                if score is not None:
                    by_type[question_type].append(float(score))
                attribution = attribute_question_failure(
                    item, full_memory=full_memory, compactor_metrics=compactor_metrics
                )
                if attribution["stage"] != "passed":
                    stage_counts[attribution["stage"]] += 1
                    failures.append({
                        "question_type": question_type,
                        "item_index": index,
                        "question": item.get("question"),
                        "score": score,
                        **attribution,
                    })

    all_scores = [score for scores in by_type.values() for score in scores]
    return {
        "runs": len(runs),
        "questions": len(all_scores),
        "judge_score_mean": round(mean(all_scores), 6) if all_scores else None,
        "judge_score_stddev": round(pstdev(all_scores), 6) if len(all_scores) > 1 else 0.0,
        "question_types": {
            name: {
                "mean": round(mean(scores), 6),
                "stddev": round(pstdev(scores), 6) if len(scores) > 1 else 0.0,
                "count": len(scores),
            }
            for name, scores in sorted(by_type.items())
        },
        "tokens": dict(sorted(token_totals.items())),
        "updater_calls": updater_calls,
        "compactor_calls": compactor_calls,
        "active_entries_mean": round(mean(active_entries), 3) if active_entries else 0.0,
        "memory_chars_mean": round(mean(memory_chars), 3) if memory_chars else 0.0,
        "structured_latency_seconds_mean": round(mean(elapsed), 6) if elapsed else 0.0,
        "selected_memory_ids_mean": round(mean(selected_counts), 3) if selected_counts else 0.0,
        "selected_memory_ids_max": max(selected_counts, default=0),
        "failure_stage_counts": {stage: stage_counts.get(stage, 0) for stage in FAILURE_STAGES},
        "failures": failures,
    }


def compare_aggregates(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    def delta(key: str) -> float | None:
        left, right = baseline.get(key), candidate.get(key)
        if left is None or right is None:
            return None
        return round(float(right) - float(left), 6)

    types = sorted(set(baseline.get("question_types", {})) | set(candidate.get("question_types", {})))
    return {
        "judge_score_mean_delta": delta("judge_score_mean"),
        "active_entries_mean_delta": delta("active_entries_mean"),
        "memory_chars_mean_delta": delta("memory_chars_mean"),
        "structured_latency_seconds_mean_delta": delta("structured_latency_seconds_mean"),
        "selected_memory_ids_mean_delta": delta("selected_memory_ids_mean"),
        "updater_calls_delta": delta("updater_calls"),
        "compactor_calls_delta": delta("compactor_calls"),
        "tokens": {
            key: int(candidate.get("tokens", {}).get(key, 0))
            - int(baseline.get("tokens", {}).get(key, 0))
            for key in sorted(set(baseline.get("tokens", {})) | set(candidate.get("tokens", {})))
        },
        "question_types": {
            name: round(
                float(candidate.get("question_types", {}).get(name, {}).get("mean", 0.0))
                - float(baseline.get("question_types", {}).get(name, {}).get("mean", 0.0)),
                6,
            )
            for name in types
        },
    }

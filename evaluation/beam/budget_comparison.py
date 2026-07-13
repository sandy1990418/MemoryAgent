"""Deterministic, paired quality/cost comparison at fixed token budgets."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def compare_fixed_budget_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        budget = int(row["context_budget_tokens"])
        if budget <= 0:
            raise ValueError("context_budget_tokens must be positive")
        groups[(budget, str(row["case_id"]), str(row["question_id"]))].append(row)
    variants = {str(row["variant"]) for row in rows}
    output = []
    for (budget, case_id, question_id), paired in sorted(groups.items()):
        present = {str(row["variant"]) for row in paired}
        if present != variants or len(paired) != len(variants):
            raise ValueError("fixed-budget comparison requires the same cases, questions, and variants")
        for row in sorted(paired, key=lambda item: str(item["variant"])):
            production_tokens = sum(int(row.get("tokens", {}).get(role, 0)) for role in ("updater", "compactor", "agent"))
            quality = float(row.get("quality", 0.0))
            output.append({
                "budget": budget, "case_id": case_id, "question_id": question_id,
                "variant": str(row["variant"]), "quality": quality,
                "production_tokens": production_tokens,
                "judge_tokens": int(row.get("tokens", {}).get("judge", 0)),
                "quality_per_1k_production_tokens": quality * 1000 / production_tokens if production_tokens else None,
                "budget_violation": int(row.get("actual_context_tokens", 0)) > budget,
                "validated_profile": "chat", "agent_memory_validated": False,
            })
    return output

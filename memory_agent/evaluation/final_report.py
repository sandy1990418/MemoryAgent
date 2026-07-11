"""Mandatory evaluation report schema and availability helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from math import isclose


REQUIRED_METRIC_SECTIONS = (
    "routing", "quality", "updater", "injection", "compactor",
    "holdout", "adversarial",
)
REQUIRED_FAILURE_SECTIONS = (
    "routing", "memory_write", "update_selection", "answer_selection", "compactor",
)
REQUIRED_TOP_LEVEL = (
    "improved_cases", "regressed_cases", "failures", "tokens", "offline_ingestion",
)
REQUIRED_INNER_FIELDS = {
    "quality": {"canonical", "incomplete", "duplicate", "stale", "raw_request",
                "active_conflict", "section_mismatch", "future_usefulness"},
    "updater": {"development", "holdout"},
    "injection": {"average", "p50", "p95", "max", "cumulative", "zero_injection_turns"},
    "compactor": {"attempted", "successful", "failed", "rejected", "skipped",
                  "before_count", "after_count", "tokens", "failure_reasons"},
    "holdout": {"passed"},
    "adversarial": {"pass_rate"},
}


def build_paired_routing_result(
    *, production_score: float | None, oracle_score: float | None,
    paired_rubric_denominator: int, sample_size: int,
    abilities: dict[str, dict[str, Any]], routing_failures: list[dict[str, Any]] | None = None,
    provider_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if paired_rubric_denominator <= 0 or sample_size <= 0:
        raise ValueError("paired denominator and sample size must be positive")
    tolerance = max(0.02, 1 / paired_rubric_denominator)
    failures = list(provider_failures or [])
    validation_gap = bool(failures) or production_score is None or oracle_score is None
    if not validation_gap:
        assert production_score is not None and oracle_score is not None
        gap = oracle_score - production_score
    else:
        gap = None
    normalized = {}
    for name, values in abilities.items():
        production = values.get("production_score")
        oracle = values.get("oracle_score")
        if production is None or oracle is None:
            status = "validation_gap"
        elif production > oracle:
            status = "improved"
        elif production < oracle:
            status = "regressed"
        else:
            status = "unchanged"
        normalized[name] = {**values, "status": status}
    return {
        "production_score": production_score, "oracle_score": oracle_score, "gap": gap,
        "paired_rubric_denominator": paired_rubric_denominator,
        "production_tolerance": tolerance, "sample_size": sample_size,
        "abilities": normalized, "routing_failures": list(routing_failures or []),
        "provider_failures": failures, "validation_gap": validation_gap,
        "production_passed": None if validation_gap else production_score + tolerance >= oracle_score,
        "oracle_cannot_offset_production": True,
    }


def validate_paired_routing_result(result: dict[str, Any]) -> None:
    required = {"production_score", "oracle_score", "gap", "paired_rubric_denominator",
                "production_tolerance", "sample_size", "abilities", "routing_failures",
                "provider_failures", "validation_gap", "production_passed",
                "oracle_cannot_offset_production"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"paired routing result missing: {sorted(missing)}")
    expected = max(0.02, 1 / int(result["paired_rubric_denominator"]))
    if not isclose(float(result["production_tolerance"]), expected):
        raise ValueError("production tolerance does not match frozen denominator")
    if result["provider_failures"] and not result["validation_gap"]:
        raise ValueError("provider failures must create a validation gap")
    if result["validation_gap"] and result["production_passed"] is not None:
        raise ValueError("validation gap cannot produce a product-score verdict")


def validate_final_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != "production-memory-report/v1":
        raise ValueError("unsupported final report schema")
    for name in REQUIRED_TOP_LEVEL:
        if name not in report:
            raise ValueError(f"final report missing {name}")
    for side in ("baseline", "candidate"):
        sections = report.get(side, {})
        for name in REQUIRED_METRIC_SECTIONS:
            if name not in sections:
                raise ValueError(f"{side} missing {name}")
            value = sections[name]
            _validate_available_or_reason(value, f"{side}.{name}")
            if (isinstance(value, dict) and value.get("status") != "unavailable"
                    and name in REQUIRED_INNER_FIELDS):
                missing = REQUIRED_INNER_FIELDS[name] - value.keys()
                if missing:
                    raise ValueError(f"{side}.{name} missing {sorted(missing)}")
    for name in REQUIRED_FAILURE_SECTIONS:
        if name not in report.get("failures", {}):
            raise ValueError(f"failures missing {name}")
    routing = report["candidate"]["routing"]
    if not (isinstance(routing, dict) and routing.get("status") == "unavailable"):
        validate_paired_routing_result(routing)
    baseline_routing = report["baseline"]["routing"]
    if not (isinstance(baseline_routing, dict) and baseline_routing.get("status") == "unavailable"):
        validate_paired_routing_result(baseline_routing)
    for name, value in report["failures"].items():
        _validate_available_or_reason(value, f"failures.{name}")
    if set(report["tokens"]) != {"estimates", "provider_usage"}:
        raise ValueError("tokens must separate estimates and provider_usage")
    _validate_available_or_reason(report["tokens"]["estimates"], "tokens.estimates")
    _validate_available_or_reason(report["tokens"]["provider_usage"], "tokens.provider_usage")
    _validate_available_or_reason(report["offline_ingestion"], "offline_ingestion")


def _validate_available_or_reason(value: Any, path: str) -> None:
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path} must be an object or list")
    if isinstance(value, dict) and value.get("status") == "unavailable":
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise ValueError(f"{path} unavailable without reason")
        return
    if isinstance(value, dict) and "status" in value and value["status"] == "validation_gap":
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise ValueError(f"{path} validation_gap without reason")


def unavailable(reason: str) -> dict[str, str]:
    if not reason.strip():
        raise ValueError("unavailable metrics require a non-empty reason")
    return {"status": "unavailable", "reason": reason}


def build_final_report(
    *,
    baseline: dict[str, Any] | None = None,
    candidate: dict[str, Any] | None = None,
    improved_cases: list[str] | None = None,
    regressed_cases: list[str] | None = None,
    failures: dict[str, Any] | None = None,
    token_estimates: dict[str, Any] | None = None,
    provider_usage: dict[str, Any] | None = None,
    offline_ingestion: dict[str, Any] | None = None,
    unavailable_reason: str = "metric was not collected in this run",
) -> dict[str, Any]:
    """Build a complete report without inventing unavailable live values."""
    missing = unavailable(unavailable_reason)

    def metrics(value: dict[str, Any] | None) -> dict[str, Any]:
        supplied = deepcopy(value or {})
        return {
            name: supplied.get(name, deepcopy(missing))
            for name in REQUIRED_METRIC_SECTIONS
        }

    supplied_failures = deepcopy(failures or {})
    report = {
        "schema_version": "production-memory-report/v1",
        "baseline": metrics(baseline),
        "candidate": metrics(candidate),
        "improved_cases": list(improved_cases or []),
        "regressed_cases": list(regressed_cases or []),
        "failures": {
            name: supplied_failures.get(name, deepcopy(missing))
            for name in REQUIRED_FAILURE_SECTIONS
        },
        "tokens": {
            "estimates": deepcopy(token_estimates) if token_estimates is not None else deepcopy(missing),
            "provider_usage": deepcopy(provider_usage) if provider_usage is not None else deepcopy(missing),
        },
        "offline_ingestion": (
            deepcopy(offline_ingestion) if offline_ingestion is not None else deepcopy(missing)
        ),
    }
    validate_final_report(report)
    return report

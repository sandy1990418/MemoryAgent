"""Regenerate frozen offline evaluation evidence without provider calls."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_agent.evaluation.final_report import build_final_report, unavailable  # noqa: E402
from memory_agent.evaluation.online_simulation import OnlineSimulation, TranscriptExchange  # noqa: E402
from memory_agent.evaluation.manifest import build_frozen_manifest, content_hash  # noqa: E402
from memory_agent.evaluation.update_selection import update_selection_metrics  # noqa: E402
from memory_agent.models.memory import MemoryEntry  # noqa: E402
from memory_agent.models.policy import CHAT_POLICY  # noqa: E402
from memory_agent.models.sections import CHAT_SECTIONS  # noqa: E402
from memory_agent.models.transcript import Turn  # noqa: E402
from memory_agent.structured.answer_context import AnswerContextConfig  # noqa: E402
from memory_agent.structured.memory import Memory  # noqa: E402
from memory_agent.structured.selector import MemorySelector  # noqa: E402
from memory_agent.structured.update_selector import UpdateMemorySelector  # noqa: E402
from memory_agent.structured.updater import MemoryUpdater  # noqa: E402

OUTPUT = ROOT / "evaluation" / "artifacts"
MATRIX = {
    "development": {
        "turn": "Please keep dark mode enabled",
        "entries": {"D1": "User prefers dark mode", "D2": "User lives in Taipei"},
        "expected": {"D1"},
    },
    "holdout": {
        "turn": "The project latency changed to 90 ms",
        "entries": {"D1": "Project latency is 80 ms", "D2": "User prefers tea"},
        "expected": {"D1"},
    },
}
ADVERSARIAL_CASES = (
    {"id": "adv-dark-mode-unrelated-location", "turn": "Please keep dark mode enabled", "entries": {"D1": "User prefers dark mode", "D2": "User lives in Taipei"}, "expected": {"D1"}},
    {"id": "adv-latency-unrelated-preference", "turn": "The project latency changed to 90 ms", "entries": {"D1": "Project latency is 80 ms", "D2": "User prefers tea"}, "expected": {"D1"}},
    {"id": "adv-dashboard-unrelated-auth", "turn": "The dashboard is now complete", "entries": {"D1": "Dashboard implementation is in progress", "D2": "Auth uses OAuth"}, "expected": {"D1"}},
    {"id": "adv-tea-unrelated-latency", "turn": "I now prefer coffee instead of tea", "entries": {"D1": "User prefers tea", "D2": "API latency is 80 ms"}, "expected": {"D1"}},
)


class _NoCallLLM:
    def complete(self, system: str, messages: list[dict[str, str]], model: str | None = None) -> str:
        raise AssertionError("deterministic simulation must not call an LLM")


def _run_selector_case(fixture: dict[str, Any]) -> tuple[list[str], int, bool, bool]:
    memory = Memory()
    for entry_id, text in fixture["entries"].items():
        memory.entries[entry_id] = MemoryEntry(entry_id, "facts", text, [1])
    selection = UpdateMemorySelector(memory).select_for_update(
        [Turn(id=2, role="user", content=fixture["turn"])], budget=128,
    )
    selected = {entry.id for entry in selection.entries}
    return (sorted(selected), selection.visible_tokens, selected == fixture["expected"],
            selection.fallback_used)


def _adversarial_report() -> dict[str, Any]:
    cases = []
    for fixture in ADVERSARIAL_CASES:
        selected, _tokens, passed, _fallback = _run_selector_case(fixture)
        cases.append({"id": fixture["id"], "selected_ids": selected,
                      "expected_ids": sorted(fixture["expected"]), "passed": passed})
    passes = sum(case["passed"] for case in cases)
    return {"cases": cases, "passes": passes, "total": len(cases),
            "pass_rate": passes / len(cases), "execution_hash": content_hash(cases),
            "evidence_kind": "executed-production-selector"}


def _online_injection_report() -> dict[str, Any]:
    memory = Memory(CHAT_SECTIONS, policy=CHAT_POLICY)
    memory.entries["M1"] = MemoryEntry("M1", "preferences", "User prefers dark mode", [1])
    report = OnlineSimulation(
        memory=memory,
        updater=MemoryUpdater(_NoCallLLM(), CHAT_SECTIONS, policy=CHAT_POLICY),
        answer_context_config=AnswerContextConfig(MemorySelector(policy=CHAT_POLICY)),
        answer_memory_budget=100, max_window_tokens=10_000,
    ).run((TranscriptExchange("Keep dark mode enabled", "Okay."),
           TranscriptExchange("What display mode do I prefer?", "Dark mode.")))
    metric = report["injection"]
    return {"average": metric["average_tokens"], "p50": metric["p50_tokens"],
            "p95": metric["p95_tokens"], "max": metric["max_tokens"],
            "cumulative": metric["cumulative_tokens"],
            "zero_injection_turns": metric["zero_injection_turns"],
            "mode": report["mode"], "answer_calls": report["answer_calls"],
            "source": "deterministic-online-simulation",
            "estimator_policy": metric["estimator_policy"]}


def _config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _selection_report() -> dict[str, Any]:
    suites: dict[str, Any] = {}
    for name, fixture in MATRIX.items():
        selected_ids, visible_tokens, _passed, fallback_used = _run_selector_case(fixture)
        selected = set(selected_ids)
        adversarial = _adversarial_report()
        metrics = update_selection_metrics(
            expected_ids=fixture["expected"], selected_ids=selected,
            conflict_ids=fixture["expected"], fallback_uses=int(fallback_used),
            unrelated_visible_entries=len(selected - fixture["expected"]),
            adversarial_passes=adversarial["passes"], adversarial_total=adversarial["total"],
        )
        suites[name] = {
            "fixture_name": f"frozen-{name}-update-selection-v1",
            "fixture_content_hash": content_hash({**fixture, "expected": sorted(fixture["expected"])}),
            "selected_ids": sorted(selected), "visible_tokens": visible_tokens,
            "token_provenance": {"kind": "estimate", "policy": "characters_divided_by_four",
                                 "provider": None},
            **metrics.as_report(),
        }
        if not metrics.passed:
            raise RuntimeError(f"{name} update-selection gate failed")
    return suites


def main() -> None:
    product = _config(ROOT / "configs" / "product.yaml")
    beam = _config(ROOT / "configs" / "beam.yaml")
    dataset_root = ROOT / str(beam.get("data_path", "BEAM/chats/100K/1"))
    dataset_reason = None
    cases: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    if dataset_root.exists():
        chat = dataset_root / "chat.json"
        probe = dataset_root / "probing_questions" / "probing_questions.json"
        if chat.exists():
            cases.append({"id": dataset_root.name, "chat": json.loads(chat.read_text())})
        if probe.exists():
            raw = json.loads(probe.read_text())
            probes = [{"id": f"{dataset_root.name}:{index}", "probe": value}
                      for index, value in enumerate(raw)]
    if not cases or not probes:
        dataset_reason = "configured BEAM chat/probe files are unavailable or incomplete"
        cases = cases or [{"id": "unavailable", "reason": dataset_reason}]
        probes = probes or [{"id": "unavailable", "reason": dataset_reason}]

    manifest = build_frozen_manifest(
        repo=ROOT, resolved_configs={"product": product, "beam": beam},
        resolved_models={
            "answer": os.getenv("BEAM_ANSWER_MODEL", "gpt-5.4-nano"),
            "memory": os.getenv("BEAM_MEMORY_MODEL", product.get("memory_model")),
            "judge": os.getenv("BEAM_JUDGE_MODEL", "gpt-5.4-nano"),
        }, dataset={"id": "BEAM/100K", "availability": "unavailable" if dataset_reason else "available",
                    "reason": dataset_reason}, cases=[
            {"id": "synthetic-development-update-selection-v1", **MATRIX["development"],
             "expected": sorted(MATRIX["development"]["expected"])},
            {"id": "synthetic-holdout-update-selection-v1", **MATRIX["holdout"],
             "expected": sorted(MATRIX["holdout"]["expected"])},
            *[{**case, "expected": sorted(case["expected"])} for case in ADVERSARIAL_CASES],
        ], probes=probes,
        development_case_ids=["synthetic-development-update-selection-v1"],
        holdout_case_ids=["synthetic-holdout-update-selection-v1",
                          *[case["id"] for case in ADVERSARIAL_CASES]], route="production",
        token_count_provenance={"kind": "estimate", "policy": "characters_divided_by_four",
                                "provider_usage": "reported separately"},
        historical_unavailable_reason=(
            "historical 0.700 artifact used oracle-like routing and is not a production baseline"
        ),
    )
    selection = _selection_report()
    adversarial = _adversarial_report()
    credential_reason = (
        "paired paid-live production/oracle evaluation was not run because OPENAI_API_KEY "
        "was unavailable to this deterministic artifact generator"
    )
    live_gap = unavailable(credential_reason)
    candidate = {
        "routing": live_gap,
        "quality": {name: unavailable("not measured by offline artifact generation") for name in
                    ("canonical", "incomplete", "duplicate", "stale", "raw_request",
                     "active_conflict", "section_mismatch", "future_usefulness")},
        "updater": selection,
        "injection": _online_injection_report(),
        "compactor": unavailable(
            "compactor was not invoked by the deterministic online simulation"
        ),
        "holdout": {**selection["holdout"], "passed": selection["holdout"]["passed"]},
        "adversarial": adversarial,
    }
    report = build_final_report(
        baseline={}, candidate=candidate,
        failures={name: [] for name in ("routing", "memory_write", "update_selection",
                                         "answer_selection", "compactor")},
        token_estimates={"update_selection": selection,
                         "provenance": "characters_divided_by_four estimator"},
        provider_usage=live_gap,
        offline_ingestion=unavailable("offline ingestion was not executed for this artifact"),
        unavailable_reason="baseline metric was not reconstructable from production routing",
    )
    report["validation"] = {"status": "validation_gap", "reason": credential_reason,
                            "paid_live": False}
    report["manifest_content_hash"] = content_hash(manifest)
    report["report_content_hash"] = content_hash({k: v for k, v in report.items()
                                                   if k != "report_content_hash"})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase0-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "final-candidate-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

from evaluation.memory.update_selection import update_selection_metrics
from memory_agent.core.models import MemoryEntry
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.update.selector import UpdateMemorySelector
from scripts.build_evaluation_artifacts import _adversarial_report

def test_update_selection_metrics_cover_bounded_recency_context():
    memory = Memory()
    for entry_id, text in (
        ("F1", "Project latency is 80 ms"),
        ("F2", "User prefers tea"),
    ):
        memory.entries[entry_id] = MemoryEntry(entry_id, "facts", text, [1])
    selection = UpdateMemorySelector(memory).select_for_update(
        [Turn(id=2, role="user", content="Latency changed to 90 ms")], budget=128,
    )
    selected_ids = {entry.id for entry in selection.entries}
    expected_ids = {"F1", "F2"}
    report = update_selection_metrics(
        expected_ids=expected_ids,
        selected_ids=selected_ids,
        conflict_ids=expected_ids,
        fallback_uses=0,
        unrelated_visible_entries=0,
        adversarial_passes=4,
        adversarial_total=4,
    )
    assert report.recall == 1.0 and report.precision == 1.0
    assert report.missed_conflicts == 0
    assert report.adversarial_pass_rate == 1.0 and report.passed
    assert selection.visible_tokens > 0
    assert UpdateMemorySelector.__module__ == "memory_agent.update.selector"


def test_adversarial_counts_and_hash_are_derived_from_executed_named_cases():
    report = _adversarial_report()
    assert report["total"] == len(report["cases"]) == 4
    assert report["passes"] == sum(case["passed"] for case in report["cases"])
    assert all(case["id"].startswith("adv-") for case in report["cases"])
    assert len(report["execution_hash"]) == 64

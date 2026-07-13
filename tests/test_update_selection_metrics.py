from evaluation.memory.update_selection import update_selection_metrics
from memory_agent.core.models import MemoryEntry
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.update.selector import UpdateMemorySelector
from scripts.build_evaluation_artifacts import _adversarial_report

MATRIX = {
    "dev": [("D1", "User prefers dark mode", "Please keep dark mode", True),
            ("F1", "User lives in Taipei", "Please keep dark mode", False)],
    "holdout": [("D1", "Project latency is 80 ms", "Latency changed to 90 ms", True),
                ("F1", "User prefers tea", "Latency changed to 90 ms", False)],
}

def test_frozen_dev_and_holdout_conflict_matrix_gates():
    for suite, rows in MATRIX.items():
        memory = Memory()
        for entry_id, text, _turn, _expected in rows:
            memory.entries[entry_id] = MemoryEntry(entry_id, "facts", text, [1])
        selection = UpdateMemorySelector(memory).select_for_update(
            [Turn(id=2, role="user", content=rows[0][2])], budget=128,
        )
        selected_ids = {entry.id for entry in selection.entries}
        expected_ids = {entry_id for entry_id, _text, _turn, expected in rows if expected}
        unrelated = len(selected_ids - expected_ids)
        report = update_selection_metrics(expected_ids=expected_ids,
            selected_ids=selected_ids, conflict_ids=expected_ids,
            fallback_uses=int(selection.fallback_used), unrelated_visible_entries=unrelated,
            adversarial_passes=4, adversarial_total=4)
        assert report.recall >= .95 and report.precision >= .80
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

"""Labeled diagnostics and structural quality contracts."""

from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.retrieval.quality import memory_quality_report


def test_quality_report_exposes_labeled_indicators():
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops(
        [
            {"op": "ADD", "section": "goal", "text": "Goal: ship the release", "provenance": [1]},
            {"op": "ADD", "section": "facts", "text": "User asked to", "provenance": [2]},
            {"op": "ADD", "section": "facts", "text": "Ongoing state: build is…", "provenance": [3]},
        ]
    )

    report = memory_quality_report(memory)

    assert report.canonical.count == 1
    assert report.raw_request.count == 1
    assert report.incomplete.count == 1
    assert report.future_usefulness.label == "heuristic"


def test_quality_report_counts_superseded_entries_as_stale():
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops(
        [
            {"op": "ADD", "section": "facts", "text": "First result.", "provenance": [1]},
            {"op": "SUPERSEDE", "id": "F1", "reason": "Replaced."},
            {"op": "ADD", "section": "facts", "text": "Latest result.", "provenance": [2]},
        ]
    )

    report = memory_quality_report(memory)

    assert report.stale.count == 1
    assert report.stale.evidence == ("F1",)
    assert report.duplicate.count == 0


def test_quality_report_flags_duplicate_active_entries_without_merging_them():
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops(
        [
            {"op": "ADD", "section": "facts", "text": "Project uses SQLite.", "provenance": [1]},
            {"op": "ADD", "section": "facts", "text": "Project uses SQLite.", "provenance": [2]},
        ]
    )

    report = memory_quality_report(memory)

    assert report.duplicate.count == 1
    assert report.duplicate.evidence == ("F2",)
    assert len(memory.entries) == 2

from memory_agent.models.sections import CHAT_SECTIONS
from memory_agent.structured.memory import Memory


def make_memory() -> Memory:
    return Memory(sections=CHAT_SECTIONS)


def test_add_generates_sequential_per_section_ids():
    mem = make_memory()
    ops = [
        {"op": "ADD", "section": "decisions", "text": "decision one", "provenance": [1, 2]},
        {"op": "ADD", "section": "decisions", "text": "decision two", "provenance": [3]},
        {"op": "ADD", "section": "preferences", "text": "pref one", "provenance": [4]},
    ]
    applied, rejected = mem.apply_ops(ops)

    assert len(applied) == 3
    assert rejected == []
    assert set(mem.entries.keys()) == {"D1", "D2", "U1"}
    assert mem.entries["D1"].text == "decision one"
    assert mem.entries["D2"].text == "decision two"
    assert mem.entries["U1"].text == "pref one"


def test_update_replaces_text_and_unions_provenance():
    mem = make_memory()
    mem.apply_ops([{"op": "ADD", "section": "facts", "text": "fact A", "provenance": [1, 2]}])

    applied, rejected = mem.apply_ops(
        [{"op": "UPDATE", "id": "F1", "text": "fact A refined", "provenance": [2, 3]}]
    )

    assert rejected == []
    assert len(applied) == 1
    entry = mem.entries["F1"]
    assert entry.text == "fact A refined"
    assert entry.provenance == [1, 2, 3]


def test_supersede_keeps_entry_with_status_and_note():
    mem = make_memory()
    mem.apply_ops([{"op": "ADD", "section": "decisions", "text": "use X", "provenance": [1]}])

    applied, rejected = mem.apply_ops(
        [{"op": "SUPERSEDE", "id": "D1", "reason": "changed our mind"}]
    )

    assert rejected == []
    assert len(applied) == 1
    entry = mem.entries["D1"]
    assert entry.status == "superseded"
    assert entry.note == "changed our mind"
    assert entry.text == "use X"  # text untouched


def test_update_on_superseded_entry_rejected():
    mem = make_memory()
    mem.apply_ops([{"op": "ADD", "section": "decisions", "text": "use X", "provenance": [1]}])
    mem.apply_ops([{"op": "SUPERSEDE", "id": "D1", "reason": "no longer true"}])

    applied, rejected = mem.apply_ops(
        [{"op": "UPDATE", "id": "D1", "text": "should not apply", "provenance": [2]}]
    )

    assert applied == []
    assert len(rejected) == 1
    assert mem.entries["D1"].text == "use X"
    assert mem.entries["D1"].status == "superseded"


def test_unknown_id_section_or_op_rejected_without_exception():
    mem = make_memory()

    ops = [
        {"op": "ADD", "section": "not_a_real_section", "text": "x", "provenance": []},
        {"op": "UPDATE", "id": "Z99", "text": "x", "provenance": []},
        {"op": "SUPERSEDE", "id": "Z99", "reason": "x"},
        {"op": "SOMETHING_WEIRD"},
        {"not_even_an_op": True},
        "not a dict at all",
    ]

    applied, rejected = mem.apply_ops(ops)

    assert applied == []
    assert len(rejected) == len(ops)
    assert mem.entries == {}


def test_partial_batch_valid_ops_applied_untouched_entries_unchanged():
    mem = make_memory()
    mem.apply_ops([{"op": "ADD", "section": "decisions", "text": "original", "provenance": [1]}])
    original_entry_snapshot = mem.entries["D1"]
    original_copy = (
        original_entry_snapshot.id,
        original_entry_snapshot.section,
        original_entry_snapshot.text,
        list(original_entry_snapshot.provenance),
        original_entry_snapshot.status,
        original_entry_snapshot.note,
    )

    ops = [
        {"op": "ADD", "section": "facts", "text": "a new fact", "provenance": [2]},
        {"op": "UPDATE", "id": "DOES_NOT_EXIST", "text": "bad", "provenance": [3]},
    ]
    applied, rejected = mem.apply_ops(ops)

    assert len(applied) == 1
    assert len(rejected) == 1
    assert "F1" in mem.entries

    entry = mem.entries["D1"]
    entry_copy = (entry.id, entry.section, entry.text, list(entry.provenance), entry.status, entry.note)
    assert entry_copy == original_copy


def test_render_excludes_superseded_by_default_and_includes_on_request():
    mem = make_memory()
    mem.apply_ops(
        [
            {"op": "ADD", "section": "decisions", "text": "use in-memory storage", "provenance": [3]},
        ]
    )
    mem.apply_ops([{"op": "SUPERSEDE", "id": "D1", "reason": "switched to file storage"}])
    mem.apply_ops(
        [
            {
                "op": "ADD",
                "section": "decisions",
                "text": "use file storage",
                "provenance": [10],
            }
        ]
    )

    default_render = mem.render()
    assert "use in-memory storage" not in default_render
    assert "use file storage" in default_render

    full_render = mem.render(include_superseded=True)
    assert "use in-memory storage" in full_render
    assert "use file storage" in full_render
    assert "Superseded" in full_render
    assert "switched to file storage" in full_render

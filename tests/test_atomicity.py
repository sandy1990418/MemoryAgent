from memory_agent.application.session import MemorySession
from memory_agent.application.structured_service import StructuredMemoryService
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


class Toggle:
    """Mutable holder so we can flip the updater's behavior mid-test."""

    def __init__(self):
        self.should_fail = True


def make_session(toggle: Toggle) -> MemorySession:
    chat_llm = ScriptedLLM(lambda system, messages: "ok")

    def updater_script(system, messages):
        if toggle.should_fail:
            raise RuntimeError("simulated updater outage")
        return '[{"op": "NOOP"}]'

    updater_llm = ScriptedLLM(updater_script)
    updater = MemoryUpdater(llm=updater_llm, sections=CHAT_SECTIONS)

    return MemorySession(
        chat_llm=chat_llm,
        updater=updater,
        sections=CHAT_SECTIONS,
        max_window_tokens=50,
    )


def test_updater_failure_never_loses_turns():
    toggle = Toggle()
    session = make_session(toggle)

    for i in range(30):
        session.send(f"this is message number {i} with some extra padding text to add tokens")

    total_turns_sent = len(session.transcript)
    assert total_turns_sent == 30 * 2  # user + assistant per send

    # Nothing was evicted since updates always failed: the window must still
    # contain the turns that would have been evicted.
    window_ids = {t.id for t in session.window.turns()}
    transcript_ids = {t.id for t in session.transcript.all()}
    assert window_ids == transcript_ids
    assert len(transcript_ids) == total_turns_sent

    # Now let updates succeed.
    toggle.should_fail = False
    session.send("final message that should trigger a successful eviction")

    window_ids_after = {t.id for t in session.window.turns()}
    transcript_ids_after = {t.id for t in session.transcript.all()}

    # Transcript is append-only and still has everything.
    assert transcript_ids.issubset(transcript_ids_after)
    # Window shrank: some turns got evicted successfully.
    assert len(window_ids_after) < len(transcript_ids_after)


def test_all_ops_rejected_keeps_turns():
    """Parseable ops that are all rejected must not evict the turns:
    nothing reached memory, so removal would silently lose information.
    """
    toggle = Toggle()
    toggle.should_fail = False
    chat_llm = ScriptedLLM(lambda system, messages: "ok")

    def updater_script(system, messages):
        if toggle.should_fail:
            return '[{"op": "NOOP"}]'
        # Valid JSON, but every op references a nonexistent entry id.
        return '[{"op": "UPDATE", "id": "D999", "text": "x", "provenance": [1]}]'

    updater = MemoryUpdater(llm=ScriptedLLM(updater_script), sections=CHAT_SECTIONS)
    session = MemorySession(
        chat_llm=chat_llm,
        updater=updater,
        sections=CHAT_SECTIONS,
        max_window_tokens=50,
    )

    for i in range(30):
        session.send(f"this is message number {i} with some extra padding text to add tokens")

    # All ops rejected every time: no turn may have been dropped.
    window_ids = {t.id for t in session.window.turns()}
    transcript_ids = {t.id for t in session.transcript.all()}
    assert window_ids == transcript_ids

    # Once the updater emits an applicable op, eviction proceeds.
    toggle.should_fail = True
    session.send("final message that should trigger a successful eviction")
    assert len(session.window.turns()) < len(session.transcript.all())


def test_partial_rejected_ops_keep_turns_and_do_not_mutate_memory():
    chat_llm = ScriptedLLM(lambda system, messages: "ok")

    def updater_script(system, messages):
        return (
            '[{"op": "ADD", "section": "decisions", "text": "valid", "provenance": [1]}, '
            '{"op": "UPDATE", "id": "D999", "text": "invalid", "provenance": [1]}]'
        )

    updater = MemoryUpdater(llm=ScriptedLLM(updater_script), sections=CHAT_SECTIONS)
    session = MemorySession(
        chat_llm=chat_llm,
        updater=updater,
        sections=CHAT_SECTIONS,
        max_window_tokens=50,
    )

    for i in range(30):
        session.send(f"this is message number {i} with some extra padding text to add tokens")

    window_ids = {t.id for t in session.window.turns()}
    transcript_ids = {t.id for t in session.transcript.all()}
    assert window_ids == transcript_ids
    assert session.memory.entries == {}


def test_multi_batch_update_covers_all_turns_and_commits_once(monkeypatch):
    responses = iter(
        [
            '[{"op":"ADD","section":"facts","text":"first batch",'
            '"provenance":[1]}]',
            '[{"op":"ADD","section":"facts","text":"second batch",'
            '"provenance":[3]}]',
        ]
    )
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: next(responses)),
        sections=CHAT_SECTIONS,
        policy=CHAT_POLICY,
        max_retries=0,
        evicted_turn_token_budget=2,
        token_estimator=lambda _text: 1,
    )
    memory = Memory(sections=CHAT_SECTIONS, policy=CHAT_POLICY)
    commits = []
    original_commit = memory.commit_trial
    monkeypatch.setattr(
        memory,
        "commit_trial",
        lambda trial, revision: commits.append(revision)
        or original_commit(trial, revision),
    )
    service = StructuredMemoryService(memory=memory, updater=updater, policy=CHAT_POLICY)

    result = service.update(
        [
            Turn(1, "user", "oldest"),
            Turn(2, "user", "middle"),
            Turn(3, "user", "newest"),
        ]
    )

    assert result.committed
    assert result.rejected_ops == []
    assert commits == [0]
    assert result.diagnostics == {
        "planned_turn_ids": [1, 2, 3],
        "planned_batch_turn_ids": [[1, 2], [3]],
        "committed_turn_ids": [1, 2, 3],
        "deferred_turn_ids": [],
        "dropped_turn_ids": [],
        "status": "committed",
    }
    assert [entry.text for entry in memory.entries.values()] == [
        "first batch",
        "second batch",
    ]


def test_late_batch_rejection_commits_verified_prefix_and_defers_suffix(monkeypatch):
    responses = iter(
        [
            '[{"op":"ADD","section":"facts","text":"staged only",'
            '"provenance":[1]}]',
            '[{"op":"UPDATE","id":"F999","text":"invalid",'
            '"provenance":[3]}]',
        ]
    )
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: next(responses)),
        sections=CHAT_SECTIONS,
        policy=CHAT_POLICY,
        max_retries=0,
        evicted_turn_token_budget=2,
        token_estimator=lambda _text: 1,
    )
    memory = Memory(sections=CHAT_SECTIONS, policy=CHAT_POLICY)
    commits = []
    original_commit = memory.commit_trial
    monkeypatch.setattr(
        memory,
        "commit_trial",
        lambda trial, revision: commits.append(revision)
        or original_commit(trial, revision),
    )
    service = StructuredMemoryService(memory=memory, updater=updater, policy=CHAT_POLICY)

    result = service.update(
        [
            Turn(1, "user", "oldest"),
            Turn(2, "user", "middle"),
            Turn(3, "user", "newest"),
        ]
    )

    assert result.committed
    assert result.failure_reason == "rejected_ops"
    assert [op["text"] for op in result.applied_ops] == ["staged only"]
    assert [entry.text for entry in memory.entries.values()] == ["staged only"]
    assert commits == [0]
    assert result.diagnostics["planned_turn_ids"] == [1, 2, 3]
    assert result.diagnostics["committed_turn_ids"] == [1, 2]
    assert result.diagnostics["deferred_turn_ids"] == [3]
    assert result.diagnostics["dropped_turn_ids"] == []
    assert result.diagnostics["status"] == "partial"


def test_late_batch_transport_failure_commits_verified_prefix_and_defers_suffix():
    calls = {"count": 0}

    def script(*_):
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                '[{"op":"ADD","section":"facts","text":"saved prefix",'
                '"provenance":[1]}]'
            )
        raise RuntimeError("transport down")

    updater = MemoryUpdater(
        llm=ScriptedLLM(script),
        sections=CHAT_SECTIONS,
        policy=CHAT_POLICY,
        max_retries=0,
        evicted_turn_token_budget=2,
        token_estimator=lambda _text: 1,
    )
    memory = Memory(sections=CHAT_SECTIONS, policy=CHAT_POLICY)
    service = StructuredMemoryService(memory=memory, updater=updater, policy=CHAT_POLICY)

    result = service.update(
        [
            Turn(1, "user", "oldest"),
            Turn(2, "user", "middle"),
            Turn(3, "user", "newest"),
        ]
    )

    assert result.committed
    assert result.failure_reason.startswith("updater_failed:")
    assert [entry.text for entry in memory.entries.values()] == ["saved prefix"]
    assert result.diagnostics["committed_turn_ids"] == [1, 2]
    assert result.diagnostics["deferred_turn_ids"] == [3]
    assert result.diagnostics["dropped_turn_ids"] == []
    assert result.diagnostics["status"] == "partial"


def test_oversized_exchange_is_one_complete_planned_batch():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: "[]"),
        sections=CHAT_SECTIONS,
        policy=CHAT_POLICY,
        evicted_turn_token_budget=1,
        token_estimator=lambda _text: 1,
    )

    batches = updater._plan_turn_batches(
        [
            Turn(1, "user", "request"),
            Turn(2, "assistant", "response"),
        ]
    )

    assert [[turn.id for turn in batch] for batch in batches] == [[1, 2]]
    assert updater.turn_selection_reports[-1]["dropped_turn_ids"] == []

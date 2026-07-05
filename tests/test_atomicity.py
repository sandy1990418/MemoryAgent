from memory_agent.sections import CHAT_SECTIONS
from memory_agent.session import MemorySession
from memory_agent.updater import MemoryUpdater
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

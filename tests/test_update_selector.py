from memory_agent.models.policy import get_memory_policy
from memory_agent.models.sections import CHAT_SECTIONS, PRACTICAL_SECTIONS
from memory_agent.models.config import ProductMemoryConfig
from memory_agent.models.transcript import Turn
from memory_agent.structured.memory import Memory
from memory_agent.structured.update_selector import UpdateMemorySelector
from memory_agent.structured.updater import MemoryUpdater
from tests.fakes import ScriptedLLM
from memory_agent.models.memory import MemoryValue, SubjectIdentity
from memory_agent.profiles.chat.subject_normalizer import ChatSubjectNormalizer


def _memory() -> Memory:
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "Deployment deadline is April 15", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "Weather API returns JSON", "provenance": [2]},
    ])
    return memory


def test_update_selector_returns_explainable_related_match():
    selection = UpdateMemorySelector(_memory(), token_estimator=lambda _text: 1).select_for_update(
        [Turn(id=3, role="user", content="Deployment deadline moved to May 1")], 2
    )
    assert [entry.id for entry in selection.entries] == ["F1"]
    assert selection.matches[0].reasons[0].startswith("lexical_overlap:")
    assert dict(selection.matches[0].score_components)["lexical_overlap"] > 0
    assert 0 < selection.matches[0].confidence <= 1


def test_update_selector_unrelated_turn_has_empty_context():
    selection = UpdateMemorySelector(_memory()).select_for_update(
        [Turn(id=3, role="user", content="Tell me a joke about penguins")], 100
    )
    assert selection.entries == ()
    assert selection.visible_tokens == 0


def test_typed_exact_subject_unit_qualifier_precedes_bounded_legacy_fallback():
    memory = _memory()
    identity = SubjectIdentity("chat", "api latency", "value", "when online", 0.9)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "When online, API latency is 250 ms.", "provenance": [3], "subject_identity": identity, "value": MemoryValue("250", "ms")},
        {"op": "ADD", "section": "facts", "text": "API latency dashboard is archived.", "provenance": [4]},
    ])
    selection = UpdateMemorySelector(memory, token_estimator=lambda _: 1,
        subject_normalizer=ChatSubjectNormalizer(), max_legacy_fallback_entries=1,
    ).select_for_update([Turn(id=5, role="user", content="When online, API latency is 200 ms.")], 5)
    assert selection.entries[0].id == "F3"
    assert selection.matches[0].reasons == ("typed_exact_subject_unit_qualifier",)
    assert selection.fallback_used is True
    assert selection.fallback_reason == "bounded_ambiguous_legacy_lexical_match"
    assert len(selection.entries) <= 2


def test_migration_on_touch_only_migrates_selected_high_confidence_legacy_entry():
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops([
        {"op":"ADD", "section":"facts", "text":"The API latency is 250 ms.", "provenance":[1]},
        {"op":"ADD", "section":"facts", "text":"The worker queue depth is 10 items.", "provenance":[2]},
    ])
    updater = MemoryUpdater(llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        sections=CHAT_SECTIONS, max_retries=0)
    updater.update(memory, [Turn(id=3, role="user", content="The API latency is 200 ms.")])
    assert memory.entries["F1"].subject_identity is not None
    assert memory.entries["F2"].subject_identity is None


def test_updater_rejects_existing_id_outside_visible_context():
    memory = _memory()
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda _system, _messages: '[{"op":"SUPERSEDE","id":"F2","reason":"hidden"}]'),
        sections=CHAT_SECTIONS,
        max_retries=0,
    )
    applied, rejected = updater.update(
        memory, [Turn(id=3, role="user", content="Deployment deadline moved")]
    )
    assert applied == []
    assert rejected[0]["reason"] == "UPDATE/SUPERSEDE id was not visible to updater"
    assert memory.entries["F2"].status == "active"


def test_update_token_report_separates_prompt_components():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda _system, _messages: '[{"op":"NOOP"}]'),
        sections=CHAT_SECTIONS,
        max_retries=0,
    )
    updater.update(_memory(), [Turn(id=3, role="user", content="Deployment deadline moved")])
    usage = updater.update_token_usage()
    assert usage["source"] == "estimator"
    assert usage["calls"] == 1
    assert usage["system_tokens"] > 0
    assert usage["visible_memory_tokens"] > 0
    assert usage["evicted_turn_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["average_tokens_per_call"] == usage["total_tokens"]


def test_evicted_turn_budget_keeps_only_complete_newest_turns():
    captured = {}
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, _messages: captured.setdefault("system", system) and '[{"op":"NOOP"}]'),
        sections=CHAT_SECTIONS,
        max_retries=0,
        evicted_turn_token_budget=2,
        token_estimator=lambda _text: 1,
    )
    updater.update(_memory(), [
        Turn(id=3, role="user", content="older turn"),
        Turn(id=4, role="user", content="newer turn"),
        Turn(id=5, role="user", content="newest turn"),
    ])
    assert "older turn" not in captured["system"]
    assert "newer turn" in captured["system"]
    assert "newest turn" in captured["system"]


def test_product_config_loads_separate_memory_budgets(tmp_path, monkeypatch):
    path = tmp_path / "product.yaml"
    path.write_text(
        "answer_memory_token_budget: 101\n"
        "update_memory_token_budget: 202\n"
        "evicted_turn_token_budget: 303\n"
    )
    monkeypatch.setenv("UPDATE_MEMORY_TOKEN_BUDGET", "222")
    config = ProductMemoryConfig.from_yaml_env(path)
    assert config.answer_memory_token_budget == 101
    assert config.update_memory_token_budget == 222
    assert config.evicted_turn_token_budget == 303


def test_product_config_loads_nested_updater_budget(tmp_path):
    path = tmp_path / "product.yaml"
    path.write_text(
        "updater:\n"
        "  max_visible_memory_tokens: 111\n"
        "  max_evicted_turn_tokens: 222\n"
        "  max_candidate_entries: 7\n"
        "  max_legacy_candidate_entries: 3\n"
    )
    config = ProductMemoryConfig.from_yaml_env(path)
    assert config.update_memory_token_budget == 111
    assert config.evicted_turn_token_budget == 222
    assert config.updater_max_candidate_entries == 7
    assert config.updater_max_legacy_candidate_entries == 3


def test_required_typed_exact_subject_reports_budget_overflow():
    memory = Memory(sections=CHAT_SECTIONS)
    identity = SubjectIdentity("chat", "api latency", "value", "when online", .9)
    memory.apply_ops([{
        "op": "ADD", "section": "facts", "text": "When online, API latency is 250 ms.",
        "provenance": [1], "subject_identity": identity, "value": MemoryValue("250", "ms"),
    }])
    selection = UpdateMemorySelector(
        memory, token_estimator=lambda _text: 10, subject_normalizer=ChatSubjectNormalizer()
    ).select_for_update([Turn(2, "user", "When online, API latency is 200 ms.")], 1)
    assert [entry.id for entry in selection.entries] == ["F1"]
    assert selection.visible_tokens == 10
    assert selection.required_overflow_tokens == 9


def test_evicted_budget_preserves_complete_multiformat_turns():
    turns = [
        Turn(1, "user", "English durable statement"),
        Turn(2, "user", "中文完整句子"),
        Turn(3, "user", "mixed 中文 JSON: {\"ok\": true}"),
        Turn(4, "assistant", "```python\nprint('完整')\n```"),
        Turn(5, "tool", "{\"rows\": [1, 2, 3]}")
    ]
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'), sections=CHAT_SECTIONS,
        evicted_turn_token_budget=3, token_estimator=lambda _text: 1,
    )
    selected = updater._turns_within_budget(turns)
    assert selected == turns[-3:]
    assert selected[0].content == "mixed 中文 JSON: {\"ok\": true}"


def test_evicted_budget_never_orphans_assistant_and_records_mandatory_overflow():
    turns = [
        Turn(1, "user", "x " * 3000),
        Turn(2, "assistant", "I will implement the complete requirement."),
    ]
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'), sections=CHAT_SECTIONS,
        evicted_turn_token_budget=1200,
    )

    assert updater._turns_within_budget(turns) == turns
    report = updater.turn_selection_reports[-1]
    assert report["selected_turn_ids"] == [1, 2]
    assert report["dropped_turn_ids"] == []
    assert report["oversized_mandatory_group"] is True
    assert report["mandatory_overflow_tokens"] > 0
    assert report["selection_is_contiguous"] is True


def test_chinese_oversized_group_and_tool_results_stay_complete():
    turns = [
        Turn(1, "user", "需求" * 3000),
        Turn(2, "assistant", "我會完整處理。"),
        Turn(3, "user", "請查詢狀態"),
        Turn(4, "assistant", "[tool_call] status({'project': 'A'})"),
        Turn(5, "tool", "running"),
        Turn(6, "tool", "healthy"),
    ]
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'), sections=CHAT_SECTIONS,
        evicted_turn_token_budget=1200,
    )

    selected = updater._turns_within_budget(turns)
    assert [turn.id for turn in selected] == [3, 4, 5, 6]
    report = updater.turn_selection_reports[-1]
    assert report["dropped_turn_ids"] == [1, 2]
    assert report["groups"][-1]["type"] == "tool_call_result"


def test_practical_budget_omits_transient_assistant_answers_but_keeps_user_sources():
    policy = get_memory_policy("chat")
    turns = [
        Turn(1, "user", "Project Alpha is blocked."),
        Turn(2, "assistant", "Long generic advice " * 200),
        Turn(3, "user", "Project Beta is active."),
        Turn(4, "assistant", "More generic advice " * 200),
    ]
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
        evicted_turn_token_budget=20,
        token_estimator=lambda text: max(1, len(text) // 20),
    )

    selected = updater._turns_within_budget(turns)

    assert [turn.id for turn in selected] == [1, 3]


def test_practical_budget_keeps_assistant_proposal_with_user_acceptance():
    policy = get_memory_policy("chat")
    turns = [
        Turn(1, "assistant", "I propose using a weekly release train."),
        Turn(2, "user", "Yes, let's do that."),
    ]
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
        evicted_turn_token_budget=1,
        token_estimator=lambda _text: 1,
    )

    assert updater._turns_within_budget(turns) == turns

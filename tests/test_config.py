"""Configuration contracts for the single chat-memory product runtime."""

import pytest

from memory_agent.application.chat import build_chat_memory
from memory_agent.models.config import ProductMemoryConfig


def test_product_config_reads_chat_runtime_limits(tmp_path, monkeypatch):
    path = tmp_path / "product.yaml"
    path.write_text(
        "compaction_threshold: 25\n"
        "memory_model: configured\n"
        "updater:\n"
        "  max_visible_memory_tokens: 700\n"
        "  max_evicted_turn_tokens: 1400\n",
        encoding="utf-8",
    )
    for name in ("MEMORY_PROFILE", "MEMORY_SECTIONS", "MEMORY_COMPACTION_THRESHOLD", "MEMORY_MODEL"):
        monkeypatch.delenv(name, raising=False)

    config = ProductMemoryConfig.from_yaml_env(path)

    assert config.compaction_threshold == 25
    assert config.memory_model == "configured"
    assert config.update_memory_token_budget == 700
    assert config.evicted_turn_token_budget == 1400
    assert not hasattr(config, "memory_profile")
    assert not hasattr(config, "sections")


def test_product_config_rejects_invalid_compaction_threshold(tmp_path, monkeypatch):
    path = tmp_path / "product.yaml"
    path.write_text("compaction_threshold: 0\n", encoding="utf-8")
    monkeypatch.delenv("MEMORY_COMPACTION_THRESHOLD", raising=False)

    with pytest.raises(ValueError, match="compaction_threshold"):
        ProductMemoryConfig.from_yaml_env(path)


def test_chat_builder_uses_the_canonical_chat_sections(tmp_path):
    path = tmp_path / "product.yaml"
    path.write_text(
        "compaction_threshold: 17\n",
        encoding="utf-8",
    )

    chat = build_chat_memory(config=ProductMemoryConfig.from_yaml_env(path), compact=False)

    assert chat.memory.sections == chat.updater.sections
    assert [section.key for section in chat.memory.sections] == [
        "decisions",
        "preferences",
        "status_changes",
        "goal",
        "facts",
        "progress",
        "open_questions",
        "failed_attempts",
    ]

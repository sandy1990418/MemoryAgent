import pytest

from memory_agent.agents.structured import build_structured_middleware
from memory_agent.models.config import (
    HybridAgentConfig,
    ProductMemoryConfig,
    StructuredAgentConfig,
)


def test_hybrid_config_defaults_to_local_mem0_for_demos(monkeypatch):
    for name in ("MEM0_BACKEND", "MEM0_DATA_DIR", "MEM0_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    config = HybridAgentConfig.from_env()

    assert config.mem0_backend == "local"
    assert config.mem0_data_dir == ".mem0"
    assert config.mem0_api_key is None


def test_hybrid_config_platform_mem0_does_not_require_data_dir(monkeypatch):
    monkeypatch.setenv("MEM0_BACKEND", "platform")
    monkeypatch.delenv("MEM0_DATA_DIR", raising=False)
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    monkeypatch.setenv("MEM0_USER_ID", "real-user")

    config = HybridAgentConfig.from_env()

    assert config.mem0_backend == "platform"
    assert config.mem0_data_dir is None
    assert config.mem0_api_key == "test-key"
    assert config.mem0_user_id == "real-user"


def test_hybrid_config_rejects_unknown_mem0_backend(monkeypatch):
    monkeypatch.setenv("MEM0_BACKEND", "other")

    with pytest.raises(ValueError, match="MEM0_BACKEND"):
        HybridAgentConfig.from_env()


def test_product_yaml_env_overrides(tmp_path, monkeypatch):
    path = tmp_path / "product.yaml"
    path.write_text(
        "memory_profile: agent\nsections: agent\n"
        "compaction_threshold: 25\nmemory_model: configured\n"
    )
    monkeypatch.setenv("MEMORY_PROFILE", "practical")
    monkeypatch.setenv("MEMORY_SECTIONS", "practical")
    monkeypatch.setenv("MEMORY_COMPACTION_THRESHOLD", "31")

    config = ProductMemoryConfig.from_yaml_env(path)

    assert config.memory_profile == "practical"
    assert config.sections == "practical"
    assert config.compaction_threshold == 31
    assert config.memory_model == "configured"


def test_structured_config_consumes_product_yaml(tmp_path, monkeypatch):
    path = tmp_path / "product.yaml"
    path.write_text(
        "memory_profile: agent\nsections: eval\n"
        "compaction_threshold: 24\nmemory_model: configured\n"
    )
    monkeypatch.delenv("MEMORY_PROFILE", raising=False)
    monkeypatch.delenv("MEMORY_SECTIONS", raising=False)
    monkeypatch.delenv("MEMORY_COMPACTION_THRESHOLD", raising=False)
    monkeypatch.delenv("MEMORY_MODEL", raising=False)

    config = StructuredAgentConfig.from_yaml_env(path)

    assert config.memory_profile == "agent"
    assert config.memory_sections == "eval"
    assert config.compact_min_active_entries == 24
    assert config.memory_model == "configured"


def test_product_config_rejects_beam_profile(tmp_path, monkeypatch):
    """"beam" is a runner-level CLI alias; the core package only knows
    practical/agent/eval (scripts normalize via normalize_beam_profile)."""
    path = tmp_path / "product.yaml"
    path.write_text("memory_profile: beam\n", encoding="utf-8")
    monkeypatch.delenv("MEMORY_PROFILE", raising=False)
    monkeypatch.delenv("MEMORY_SECTIONS", raising=False)

    with pytest.raises(ValueError, match="memory_profile"):
        ProductMemoryConfig.from_yaml_env(path)


def test_structured_builder_applies_product_sections_and_compaction_threshold():
    config = StructuredAgentConfig(
        memory_profile="agent",
        memory_sections="practical",
        compact_min_active_entries=17,
    )

    middleware = build_structured_middleware(config)

    assert {section.key for section in middleware.memory.sections} == {
        "decisions",
        "preferences",
        "status_changes",
        "goal",
        "facts",
        "open_questions",
        "failed_attempts",
    }
    assert middleware.compact_min_active_entries == 17


# BEAM runner config tests live in scripts/beam_tests/test_beam_config.py.

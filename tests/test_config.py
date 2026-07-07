import pytest

from memory_agent.models.config import HybridAgentConfig


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

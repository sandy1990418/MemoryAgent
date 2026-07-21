import pytest

from scripts.beam_models import BeamConfig


def test_beam_yaml_env_overrides(tmp_path, monkeypatch):
    path = tmp_path / "beam.yaml"
    path.write_text("data_path: BEAM/chats/100K/2\nabilities:\n  - abstention\njudge: false\nmax_questions_per_type: 3\n")
    monkeypatch.setenv("BEAM_ABILITIES", "knowledge_update,summarization")

    config = BeamConfig.from_yaml_env(path)

    assert str(config.data_path) == "BEAM/chats/100K/2"
    assert config.abilities == ("knowledge_update", "summarization")
    assert config.judge is False
    assert config.max_questions_per_type == 3


def test_beam_yaml_controls_models_and_runner_knobs(tmp_path, monkeypatch):
    for name in ("BEAM_ANSWER_MODEL", "BEAM_MEMORY_MODEL", "BEAM_JUDGE_MODEL",
                 "MEM0_LLM_MODEL", "MEMORY_MODEL", "BEAM_TOP_K"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "beam.yaml"
    path.write_text(
        "answer_model: answer-x\njudge_model: judge-x\n"
        "top_k: 4\nstructured_max_tokens: 9000\nstructured_evict_fraction: 0.25\n"
        "recursion_limit: 12\n"
    )
    monkeypatch.setenv("BEAM_TOP_K", "5")

    config = BeamConfig.from_yaml_env(path)

    assert config.answer_model == "answer-x"
    assert config.memory_model == "answer-x"  # falls back to answer_model
    assert config.judge_model == "judge-x"
    assert config.top_k == 5  # env beats yaml
    assert config.structured_max_tokens == 9000
    assert config.structured_evict_fraction == 0.25
    assert config.recursion_limit == 12

    defaults = config.to_run_defaults()
    assert defaults["answer_model"] == "answer-x"
    assert defaults["structured_model"] == "answer-x"
    assert defaults["top_k"] == 5
    assert defaults["structured_max_tokens"] == 9000
    assert defaults["recursion_limit"] == 12


def test_beam_memory_model_falls_back_to_memory_model_env(tmp_path, monkeypatch):
    for name in ("BEAM_ANSWER_MODEL", "BEAM_MEMORY_MODEL", "MEM0_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMORY_MODEL", "memory-env")
    path = tmp_path / "beam.yaml"
    path.write_text("answer_model: answer-x\n")

    config = BeamConfig.from_yaml_env(path)

    assert config.memory_model == "memory-env"


def test_beam_config_rejects_invalid_evict_fraction(tmp_path, monkeypatch):
    monkeypatch.delenv("BEAM_STRUCTURED_EVICT_FRACTION", raising=False)
    path = tmp_path / "beam.yaml"
    path.write_text("structured_evict_fraction: 1.5\n")

    with pytest.raises(ValueError, match="structured_evict_fraction"):
        BeamConfig.from_yaml_env(path)


def test_fixed_token_budgets_support_yaml_and_env(tmp_path, monkeypatch):
    path = tmp_path / "beam.yaml"
    path.write_text("fixed_token_budgets: [128, 256]\n")
    monkeypatch.setenv("BEAM_FIXED_TOKEN_BUDGETS", "64,128")
    config = BeamConfig.from_yaml_env(path)
    assert config.fixed_token_budgets == (64, 128)
    assert config.to_run_defaults()["fixed_token_budgets"] == [64, 128]


@pytest.mark.parametrize("value", ("0,128", "128,128", ""))
def test_fixed_token_budgets_reject_invalid_values(tmp_path, monkeypatch, value):
    path = tmp_path / "beam.yaml"
    path.write_text(f"fixed_token_budgets: [{value}]\n")
    monkeypatch.delenv("BEAM_FIXED_TOKEN_BUDGETS", raising=False)
    with pytest.raises(ValueError, match="unique positive"):
        BeamConfig.from_yaml_env(path)

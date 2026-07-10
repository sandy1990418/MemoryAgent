import pytest

from scripts.beam_models import BeamConfig, normalize_beam_profile


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
    assert config.mem0_llm_model == "answer-x"
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


def test_normalize_beam_profile_maps_cli_alias_onto_eval():
    assert normalize_beam_profile("beam") == "eval"
    assert normalize_beam_profile("practical") == "practical"
    assert normalize_beam_profile("agent") == "agent"
    assert normalize_beam_profile("eval") == "eval"

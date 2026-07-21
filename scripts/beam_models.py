"""Configuration and data models for the BEAM evaluation runners.

BEAM is an evaluation concern, so this module lives under scripts/ next to the
runners instead of inside the memory_agent package.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from memory_agent.models.config import (
    DEFAULT_PRODUCT_CONFIG_PATH,
    config_value,
    load_simple_yaml,
)


DEFAULT_CHAT_PATH = Path("BEAM/chats/100K/1/chat.json")
DEFAULT_PROBES_PATH = Path("BEAM/chats/100K/1/probing_questions/probing_questions.json")
DEFAULT_TOPICS_PATH = Path("BEAM/chats/100K/1/topic.json")
DEFAULT_RESULTS_DIR = Path("data/beam/results/100K/1")
DEFAULT_BEAM_MODEL = os.getenv("BEAM_ANSWER_MODEL", "gpt-5.4-nano")
DEFAULT_BEAM_MEMORY_MODEL = os.getenv(
    "BEAM_MEMORY_MODEL",
    os.getenv("MEMORY_MODEL", DEFAULT_BEAM_MODEL),
)
DEFAULT_BEAM_JUDGE_MODEL = os.getenv("BEAM_JUDGE_MODEL", DEFAULT_BEAM_MODEL)
DEFAULT_BEAM_QUESTION_TYPES = (
    "contradiction_resolution",
    "knowledge_update",
    "preference_following",
    "instruction_following",
    "abstention",
    "summarization",
)
ANSWER_MEMORY_SELECTION_MODES = ("all", "selector")
DEFAULT_BEAM_CONFIG_PATH = Path(os.getenv("BEAM_CONFIG", "configs/beam.yaml"))


@dataclass(frozen=True)
class BeamConfig:
    data_path: Path = Path("BEAM/chats/100K/1")
    abilities: tuple[str, ...] = DEFAULT_BEAM_QUESTION_TYPES
    judge: bool = True
    max_questions_per_type: int | None = None
    routing_mode: str = "production"
    answer_model: str = "gpt-5.4-nano"
    memory_model: str = "gpt-5.4-nano"
    judge_model: str = "gpt-5.4-nano"
    top_k: int = 8
    max_hit_chars: int = 6000
    max_active_context_chars: int = 12000
    structured_max_tokens: int = 12000
    structured_max_memory_tokens: int = 3000
    structured_answer_tokens: int = 4000
    answer_memory_selection: str = "all"
    structured_evict_fraction: float = 0.5
    structured_keep_messages: int = 2
    recursion_limit: int = 50
    fixed_token_budgets: tuple[int, ...] = (256, 512, 1024)

    @classmethod
    def from_yaml_env(cls, path: str | Path = "configs/beam.yaml") -> "BeamConfig":
        config_path = Path(path)
        data = load_simple_yaml(config_path) if config_path.exists() else {}
        abilities = config_value(data, "abilities", "BEAM_ABILITIES", list(cls.abilities))
        if isinstance(abilities, str):
            abilities = [item.strip() for item in abilities.split(",") if item.strip()]
        max_questions = config_value(
            data, "max_questions_per_type", "BEAM_MAX_QUESTIONS_PER_TYPE", cls.max_questions_per_type
        )
        resolved_abilities = tuple(abilities)
        if not resolved_abilities:
            raise ValueError("abilities must contain at least one BEAM question type")
        resolved_max_questions = None if max_questions is None else int(max_questions)
        if resolved_max_questions is not None and resolved_max_questions < 1:
            raise ValueError("max_questions_per_type must be at least 1")
        answer_model = str(
            config_value(data, "answer_model", "BEAM_ANSWER_MODEL", cls.answer_model)
        )
        memory_model = str(
            config_value(data, "memory_model", "BEAM_MEMORY_MODEL", None)
            or os.getenv("MEMORY_MODEL")
            or answer_model
        )
        evict_fraction = float(
            config_value(
                data,
                "structured_evict_fraction",
                "BEAM_STRUCTURED_EVICT_FRACTION",
                cls.structured_evict_fraction,
            )
        )
        if not 0 < evict_fraction <= 1:
            raise ValueError("structured_evict_fraction must be in (0, 1]")
        budgets = config_value(data, "fixed_token_budgets", "BEAM_FIXED_TOKEN_BUDGETS", list(cls.fixed_token_budgets))
        if isinstance(budgets, str):
            budgets = [item.strip() for item in budgets.split(",") if item.strip()]
        resolved_budgets = tuple(int(item) for item in budgets)
        if not resolved_budgets or any(item <= 0 for item in resolved_budgets) or len(set(resolved_budgets)) != len(resolved_budgets):
            raise ValueError("fixed_token_budgets must contain unique positive integers")
        answer_memory_selection = str(
            config_value(
                data,
                "answer_memory_selection",
                "BEAM_ANSWER_MEMORY_SELECTION",
                cls.answer_memory_selection,
            )
        )
        if answer_memory_selection not in ANSWER_MEMORY_SELECTION_MODES:
            choices = ", ".join(ANSWER_MEMORY_SELECTION_MODES)
            raise ValueError(f"answer_memory_selection must be one of: {choices}")

        def _int_value(key: str, env: str, default: int, minimum: int) -> int:
            value = int(config_value(data, key, env, default))
            if value < minimum:
                raise ValueError(f"{key} must be at least {minimum}")
            return value

        return cls(
            data_path=Path(str(config_value(data, "data_path", "BEAM_DATA_PATH", cls.data_path))),
            abilities=resolved_abilities,
            judge=bool(config_value(data, "judge", "BEAM_JUDGE", cls.judge)),
            max_questions_per_type=resolved_max_questions,
            answer_model=answer_model,
            memory_model=memory_model,
            judge_model=str(
                config_value(data, "judge_model", "BEAM_JUDGE_MODEL", answer_model)
            ),
            top_k=_int_value("top_k", "BEAM_TOP_K", cls.top_k, 1),
            max_hit_chars=_int_value(
                "max_hit_chars", "BEAM_MAX_HIT_CHARS", cls.max_hit_chars, 1
            ),
            max_active_context_chars=_int_value(
                "max_active_context_chars",
                "BEAM_MAX_ACTIVE_CONTEXT_CHARS",
                cls.max_active_context_chars,
                1,
            ),
            structured_max_tokens=_int_value(
                "structured_max_tokens",
                "BEAM_STRUCTURED_MAX_TOKENS",
                cls.structured_max_tokens,
                1,
            ),
            structured_max_memory_tokens=_int_value(
                "structured_max_memory_tokens",
                "BEAM_STRUCTURED_MAX_MEMORY_TOKENS",
                cls.structured_max_memory_tokens,
                1,
            ),
            structured_answer_tokens=_int_value(
                "structured_answer_tokens",
                "BEAM_STRUCTURED_ANSWER_TOKENS",
                cls.structured_answer_tokens,
                1,
            ),
            answer_memory_selection=answer_memory_selection,
            structured_evict_fraction=evict_fraction,
            structured_keep_messages=_int_value(
                "structured_keep_messages",
                "BEAM_STRUCTURED_KEEP_MESSAGES",
                cls.structured_keep_messages,
                0,
            ),
            recursion_limit=_int_value(
                "recursion_limit", "BEAM_RECURSION_LIMIT", cls.recursion_limit, 1
            ),
            fixed_token_budgets=resolved_budgets,
        )

    def to_run_defaults(self) -> dict[str, Any]:
        return {
            "chat": self.data_path / "chat.json",
            "probes": self.data_path / "probing_questions" / "probing_questions.json",
            "topics": self.data_path / "topic.json",
            "question_types": list(self.abilities),
            "max_questions_per_type": self.max_questions_per_type,
            "judge_model": self.judge_model if self.judge else None,
            "answer_model": self.answer_model,
            "structured_model": self.memory_model,
            "top_k": self.top_k,
            "max_hit_chars": self.max_hit_chars,
            "max_active_context_chars": self.max_active_context_chars,
            "structured_max_tokens": self.structured_max_tokens,
            "structured_max_memory_tokens": self.structured_max_memory_tokens,
            "structured_answer_tokens": self.structured_answer_tokens,
            "answer_memory_selection": self.answer_memory_selection,
            "structured_evict_fraction": self.structured_evict_fraction,
            "structured_keep_messages": self.structured_keep_messages,
            "recursion_limit": self.recursion_limit,
            "fixed_token_budgets": list(self.fixed_token_budgets),
        }


def beam_config_from_argv(
    argv: list[str] | None = None,
) -> tuple[Path, BeamConfig]:
    """Resolve the BEAM config before constructing a runner's full parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--beam-config",
        type=Path,
        default=DEFAULT_BEAM_CONFIG_PATH,
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    known, _ = parser.parse_known_args(argv)
    injected: list[str] = []
    for name, value in dotenv_values(known.env_file).items():
        if name.startswith("BEAM_") and value is not None and name not in os.environ:
            os.environ[name] = value
            injected.append(name)
    try:
        config = BeamConfig.from_yaml_env(known.beam_config)
    finally:
        for name in injected:
            os.environ.pop(name, None)
    return known.beam_config, config


@dataclass(frozen=True)
class BeamRunConfig:
    beam_config: Path | None = DEFAULT_BEAM_CONFIG_PATH
    product_config: Path | None = DEFAULT_PRODUCT_CONFIG_PATH
    chat: Path = DEFAULT_CHAT_PATH
    probes: Path = DEFAULT_PROBES_PATH
    topics: Path = DEFAULT_TOPICS_PATH
    results_dir: Path = DEFAULT_RESULTS_DIR
    output: Path | None = None
    answers_output: Path | None = None
    evaluation_output: Path | None = None
    env_file: Path = Path(".env")
    top_k: int = 8
    max_hit_chars: int = 6000
    max_active_context_chars: int = 12000
    skip_ingest: bool = False
    routing_mode: str = "production"
    answer_model: str = DEFAULT_BEAM_MODEL
    structured_model: str = DEFAULT_BEAM_MEMORY_MODEL
    structured_max_tokens: int = 12000
    structured_max_memory_tokens: int = 3000
    structured_answer_tokens: int = 4000
    answer_memory_selection: str = "all"
    structured_evict_fraction: float = 0.5
    structured_keep_messages: int = 2
    structured_flush_final: bool = True
    judge_model: str | None = DEFAULT_BEAM_JUDGE_MODEL
    question_types: list[str] | tuple[str, ...] | None = DEFAULT_BEAM_QUESTION_TYPES
    max_questions_per_type: int | None = None
    memory_snapshot_output: Path | None = None
    replay_memory: Path | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BeamRunConfig":
        return cls(**vars(args))


@dataclass(frozen=True)
class BeamDeepAgentRunConfig(BeamRunConfig):
    """DeepAgent comparison configuration, isolated from the standard runner."""
    recursion_limit: int = 50

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BeamDeepAgentRunConfig":
        return cls(**vars(args))


@dataclass(frozen=True)
class BeamChunk:
    text: str
    metadata: dict[str, Any]

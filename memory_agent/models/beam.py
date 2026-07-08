"""Data models for the BEAM smoke-test runners."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CHAT_PATH = Path("data/beam/100K/1/chat.json")
DEFAULT_PROBES_PATH = Path("data/beam/100K/1/probing_questions/probing_questions.json")
DEFAULT_TOPICS_PATH = Path("data/beam/topics/100k/100k_topics.json")
DEFAULT_RESULTS_DIR = Path("data/beam/results/100K/1")
DEFAULT_BEAM_MODEL = os.getenv("BEAM_ANSWER_MODEL", "gpt-5.4-nano")
DEFAULT_BEAM_MEMORY_MODEL = os.getenv(
    "BEAM_MEMORY_MODEL",
    os.getenv("MEMORY_MODEL", DEFAULT_BEAM_MODEL),
)
DEFAULT_BEAM_JUDGE_MODEL = os.getenv("BEAM_JUDGE_MODEL", DEFAULT_BEAM_MODEL)
DEFAULT_MEM0_LLM_MODEL = os.getenv("MEM0_LLM_MODEL", DEFAULT_BEAM_MODEL)


@dataclass(frozen=True)
class BeamRunConfig:
    chat: Path = DEFAULT_CHAT_PATH
    probes: Path = DEFAULT_PROBES_PATH
    topics: Path = DEFAULT_TOPICS_PATH
    results_dir: Path = DEFAULT_RESULTS_DIR
    store_dir: Path | None = None
    output: Path | None = None
    answers_output: Path | None = None
    evaluation_output: Path | None = None
    env_file: Path = Path(".env")
    user_id: str = "beam-100k-case-1"
    memory_mode: str = "structured_only"
    top_k: int = 8
    max_hit_chars: int = 6000
    max_active_context_chars: int = 12000
    skip_ingest: bool = False
    answer_model: str = DEFAULT_BEAM_MODEL
    structured_model: str = DEFAULT_BEAM_MEMORY_MODEL
    structured_max_tokens: int = 12000
    structured_max_memory_tokens: int = 3000
    structured_answer_tokens: int = 4000
    structured_evict_fraction: float = 0.5
    structured_keep_messages: int = 2
    structured_flush_final: bool = True
    mem0_llm_model: str = DEFAULT_MEM0_LLM_MODEL
    judge_model: str | None = DEFAULT_BEAM_JUDGE_MODEL
    question_types: list[str] | None = None
    max_questions_per_type: int | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BeamRunConfig":
        return cls(**vars(args))


@dataclass(frozen=True)
class BeamDeepAgentRunConfig(BeamRunConfig):
    # The deepagent runner defaults to structured-memory-only answering;
    # mem0-backed retrieval is opt-in via --memory-mode structured_mem0.
    memory_mode: str = "structured_only"
    recursion_limit: int = 50

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BeamDeepAgentRunConfig":
        return cls(**vars(args))


@dataclass(frozen=True)
class BeamChunk:
    text: str
    metadata: dict[str, Any]

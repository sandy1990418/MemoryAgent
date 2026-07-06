"""Configuration objects for demos and benchmark runners."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project_env(env_file: str | Path | None = None) -> None:
    """Load the project .env file using a single convention across scripts."""
    load_dotenv(Path(env_file) if env_file is not None else PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class SummaryAgentConfig:
    main_model: str = "openai:gpt-5.4-nano"
    summary_model: str = "openai:gpt-5.4-nano"
    thread_id: str = "react-summary-demo"

    @classmethod
    def from_env(cls) -> "SummaryAgentConfig":
        return cls(
            main_model=os.getenv("MAIN_MODEL", cls.main_model),
            summary_model=os.getenv("SUMMARY_MODEL", cls.summary_model),
            thread_id=os.getenv("THREAD_ID", cls.thread_id),
        )


@dataclass(frozen=True)
class StructuredAgentConfig:
    main_model: str = "openai:gpt-5.4-nano"
    memory_model: str = "openai:gpt-5.4-nano"
    thread_id: str = "react-structured-memory-demo"
    max_tokens: int = 600
    max_memory_tokens: int = 600

    @classmethod
    def from_env(cls) -> "StructuredAgentConfig":
        memory_model = os.getenv("MEMORY_MODEL", os.getenv("SUMMARY_MODEL", cls.memory_model))
        return cls(
            main_model=os.getenv("MAIN_MODEL", cls.main_model),
            memory_model=memory_model,
            thread_id=os.getenv("THREAD_ID", cls.thread_id),
            max_tokens=_env_int("STRUCTURED_MAX_TOKENS", cls.max_tokens),
            max_memory_tokens=_env_int("STRUCTURED_MAX_MEMORY_TOKENS", cls.max_memory_tokens),
        )


@dataclass(frozen=True)
class HybridAgentConfig:
    main_model: str = "openai:gpt-5.4-nano"
    memory_model: str = "openai:gpt-5.4-nano"
    thread_id: str = "react-hybrid-memory-demo"
    structured_max_tokens: int = 220
    structured_max_memory_tokens: int = 600
    mem0_user_id: str = "demo-user"
    mem0_data_dir: str = ".mem0"
    mem0_llm_model: str = "gpt-5.4-nano"

    @classmethod
    def from_env(cls) -> "HybridAgentConfig":
        memory_model = os.getenv("MEMORY_MODEL", os.getenv("SUMMARY_MODEL", cls.memory_model))
        return cls(
            main_model=os.getenv("MAIN_MODEL", cls.main_model),
            memory_model=memory_model,
            thread_id=os.getenv("THREAD_ID", cls.thread_id),
            structured_max_tokens=_env_int(
                "STRUCTURED_MAX_TOKENS",
                cls.structured_max_tokens,
            ),
            structured_max_memory_tokens=_env_int(
                "STRUCTURED_MAX_MEMORY_TOKENS",
                cls.structured_max_memory_tokens,
            ),
            mem0_user_id=os.getenv("MEM0_USER_ID", cls.mem0_user_id),
            mem0_data_dir=os.getenv("MEM0_DATA_DIR", cls.mem0_data_dir),
            mem0_llm_model=os.getenv("MEM0_LLM_MODEL", cls.mem0_llm_model),
        )


@dataclass(frozen=True)
class SessionDemoConfig:
    main_model: str = "openai:gpt-5.4-nano"
    memory_model: str = "openai:gpt-5.4-nano"
    max_window_tokens: int = 300

    @classmethod
    def from_env(cls) -> "SessionDemoConfig":
        memory_model = os.getenv("MEMORY_MODEL", os.getenv("SUMMARY_MODEL", cls.memory_model))
        return cls(
            main_model=os.getenv("MAIN_MODEL", cls.main_model),
            memory_model=memory_model,
            max_window_tokens=_env_int("MAX_WINDOW_TOKENS", cls.max_window_tokens),
        )

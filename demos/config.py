"""Configuration used only by runnable demos."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


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


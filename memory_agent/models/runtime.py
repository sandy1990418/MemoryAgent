"""Runtime container models returned by app builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StructuredAgentRuntime:
    agent: Any
    structured_middleware: Any


@dataclass(frozen=True)
class HybridAgentRuntime:
    agent: Any
    structured_middleware: Any
    long_term_middleware: Any | None

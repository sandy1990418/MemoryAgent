"""Environment-backed configuration models."""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCT_CONFIG_PATH = Path(
    os.getenv("PRODUCT_MEMORY_CONFIG", "configs/product.yaml")
)


def load_project_env(env_file: str | Path | None = None) -> None:
    """Load the project .env file using a single convention across scripts."""
    load_dotenv(Path(env_file) if env_file is not None else PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _parse_scalar(value: str):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")


def load_simple_yaml(path: str | Path) -> dict[str, object]:
    """Load the small YAML subset used by repo config files.

    Supports top-level scalars/lists and one level of scalar mappings (used by
    ``updater:``).  It intentionally remains dependency-free.
    """
    data: dict[str, object] = {}
    current_key: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            data.setdefault(current_key, [])
            assert isinstance(data[current_key], list)
            data[current_key].append(_parse_scalar(line[4:]))
            continue
        if line.startswith("  ") and current_key and ":" in line:
            child_key, child_value = line.strip().split(":", 1)
            if not isinstance(data.get(current_key), dict):
                data[current_key] = {}
            assert isinstance(data[current_key], dict)
            data[current_key][child_key.strip()] = _parse_scalar(child_value)
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current_key = key.strip()
        if value.strip():
            data[current_key] = _parse_scalar(value)
        else:
            data[current_key] = []
    return data


def config_value(data: dict[str, object], key: str, env: str, default):
    raw = os.getenv(env)
    if raw is not None and raw != "":
        return _parse_scalar(raw)
    return data.get(key, default)


@dataclass(frozen=True)
class ProductMemoryConfig:
    compaction_threshold: int = 30
    memory_model: str = "openai:gpt-5.4-nano"
    answer_memory_token_budget: int = 600
    update_memory_token_budget: int = 600
    evicted_turn_token_budget: int = 1200
    updater_max_candidate_entries: int = 8

    @classmethod
    def from_yaml_env(cls, path: str | Path = "configs/product.yaml") -> "ProductMemoryConfig":
        data = load_simple_yaml(path) if Path(path).exists() else {}
        compaction_threshold = int(
            config_value(
                data,
                "compaction_threshold",
                "MEMORY_COMPACTION_THRESHOLD",
                cls.compaction_threshold,
            )
        )
        if compaction_threshold < 1:
            raise ValueError("compaction_threshold must be at least 1")
        updater_data = data.get("updater", {})
        if not isinstance(updater_data, dict):
            updater_data = {}
        def updater_int(key: str, env: str, default: int) -> int:
            raw = os.getenv(env)
            if raw not in (None, ""):
                return int(raw)
            return int(updater_data.get(key, default))
        return cls(
            compaction_threshold=compaction_threshold,
            memory_model=str(config_value(data, "memory_model", "MEMORY_MODEL", cls.memory_model)),
            answer_memory_token_budget=int(config_value(data, "answer_memory_token_budget", "ANSWER_MEMORY_TOKEN_BUDGET", cls.answer_memory_token_budget)),
            update_memory_token_budget=updater_int("max_visible_memory_tokens", "UPDATE_MEMORY_TOKEN_BUDGET", cls.update_memory_token_budget),
            evicted_turn_token_budget=updater_int("max_evicted_turn_tokens", "EVICTED_TURN_TOKEN_BUDGET", cls.evicted_turn_token_budget),
            updater_max_candidate_entries=updater_int("max_candidate_entries", "UPDATER_MAX_CANDIDATE_ENTRIES", cls.updater_max_candidate_entries),
        )

def product_config_from_argv(
    argv: list[str] | None = None,
) -> tuple[Path, ProductMemoryConfig]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--product-config",
        type=Path,
        default=DEFAULT_PRODUCT_CONFIG_PATH,
    )
    known, _ = parser.parse_known_args(argv)
    return known.product_config, ProductMemoryConfig.from_yaml_env(known.product_config)

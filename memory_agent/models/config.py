"""Environment-backed configuration models."""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEM0_BACKENDS = frozenset({"local", "platform", "disabled"})
MEMORY_PROFILES = frozenset({"practical", "agent", "eval"})
MEMORY_SECTION_PRESETS = frozenset({"practical", "agent", "eval"})
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


def _env_optional(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw == "":
        return None
    return raw


def _env_mem0_backend(default: str) -> str:
    backend = os.getenv("MEM0_BACKEND", default).strip().lower()
    if backend not in MEM0_BACKENDS:
        choices = ", ".join(sorted(MEM0_BACKENDS))
        raise ValueError(f"MEM0_BACKEND must be one of: {choices}")
    return backend


def _env_memory_profile(default: str) -> str:
    profile = os.getenv("MEMORY_PROFILE", default).strip().lower()
    if profile not in MEMORY_PROFILES:
        choices = ", ".join(sorted(MEMORY_PROFILES))
        raise ValueError(f"MEMORY_PROFILE must be one of: {choices}")
    return profile


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
    """Load the flat YAML subset used by repo config files without a dependency."""
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
    memory_profile: str = "practical"
    sections: str = "practical"
    compaction_threshold: int = 30
    memory_model: str = "openai:gpt-5.4-nano"

    @classmethod
    def from_yaml_env(cls, path: str | Path = "configs/product.yaml") -> "ProductMemoryConfig":
        data = load_simple_yaml(path) if Path(path).exists() else {}
        profile = str(config_value(data, "memory_profile", "MEMORY_PROFILE", cls.memory_profile))
        if profile not in MEMORY_PROFILES:
            choices = ", ".join(sorted(MEMORY_PROFILES))
            raise ValueError(f"memory_profile must be one of: {choices}")
        sections = str(config_value(data, "sections", "MEMORY_SECTIONS", profile))
        if sections not in MEMORY_SECTION_PRESETS:
            choices = ", ".join(sorted(MEMORY_SECTION_PRESETS))
            raise ValueError(f"sections must be one of: {choices}")
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
        return cls(
            memory_profile=profile,
            sections=sections,
            compaction_threshold=compaction_threshold,
            memory_model=str(config_value(data, "memory_model", "MEMORY_MODEL", cls.memory_model)),
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


@dataclass(frozen=True)
class StructuredAgentConfig:
    main_model: str = "openai:gpt-5.4-nano"
    memory_model: str = "openai:gpt-5.4-nano"
    thread_id: str = "react-structured-memory-demo"
    max_tokens: int = 600
    max_memory_tokens: int = 600
    keep_messages: int = 4
    memory_profile: str = "practical"
    memory_sections: str | None = None
    compact_min_active_entries: int = 30

    @classmethod
    def from_env(cls) -> "StructuredAgentConfig":
        memory_model = os.getenv("MEMORY_MODEL", os.getenv("SUMMARY_MODEL", cls.memory_model))
        return cls(
            main_model=os.getenv("MAIN_MODEL", cls.main_model),
            memory_model=memory_model,
            thread_id=os.getenv("THREAD_ID", cls.thread_id),
            max_tokens=_env_int("STRUCTURED_MAX_TOKENS", cls.max_tokens),
            max_memory_tokens=_env_int("STRUCTURED_MAX_MEMORY_TOKENS", cls.max_memory_tokens),
            keep_messages=_env_int("STRUCTURED_KEEP_MESSAGES", cls.keep_messages),
            memory_profile=_env_memory_profile(cls.memory_profile),
            memory_sections=os.getenv("MEMORY_SECTIONS") or None,
            compact_min_active_entries=_env_int(
                "MEMORY_COMPACTION_THRESHOLD",
                cls.compact_min_active_entries,
            ),
        )

    @classmethod
    def from_yaml_env(
        cls,
        path: str | Path = "configs/product.yaml",
    ) -> "StructuredAgentConfig":
        product = ProductMemoryConfig.from_yaml_env(path)
        return cls(
            main_model=os.getenv("MAIN_MODEL", cls.main_model),
            memory_model=product.memory_model,
            thread_id=os.getenv("THREAD_ID", cls.thread_id),
            max_tokens=_env_int("STRUCTURED_MAX_TOKENS", cls.max_tokens),
            max_memory_tokens=_env_int("STRUCTURED_MAX_MEMORY_TOKENS", cls.max_memory_tokens),
            keep_messages=_env_int("STRUCTURED_KEEP_MESSAGES", cls.keep_messages),
            memory_profile=product.memory_profile,
            memory_sections=product.sections,
            compact_min_active_entries=product.compaction_threshold,
        )


@dataclass(frozen=True)
class HybridAgentConfig:
    main_model: str = "openai:gpt-5.4-nano"
    memory_model: str = "openai:gpt-5.4-nano"
    thread_id: str = "react-hybrid-memory-demo"
    structured_max_tokens: int = 220
    structured_max_memory_tokens: int = 600
    structured_keep_messages: int = 4
    mem0_backend: str = "local"
    mem0_user_id: str = "demo-user"
    mem0_data_dir: str | None = ".mem0"
    mem0_llm_model: str | None = "gpt-5.4-nano"
    mem0_api_key: str | None = None
    memory_profile: str = "practical"

    @classmethod
    def from_env(cls) -> "HybridAgentConfig":
        memory_model = os.getenv("MEMORY_MODEL", os.getenv("SUMMARY_MODEL", cls.memory_model))
        backend = _env_mem0_backend(cls.mem0_backend)
        default_data_dir = cls.mem0_data_dir if backend == "local" else None
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
            structured_keep_messages=_env_int(
                "STRUCTURED_KEEP_MESSAGES",
                cls.structured_keep_messages,
            ),
            mem0_backend=backend,
            mem0_user_id=os.getenv("MEM0_USER_ID", cls.mem0_user_id),
            mem0_data_dir=_env_optional("MEM0_DATA_DIR", default_data_dir),
            mem0_llm_model=_env_optional("MEM0_LLM_MODEL", cls.mem0_llm_model),
            mem0_api_key=_env_optional("MEM0_API_KEY"),
            memory_profile=_env_memory_profile(cls.memory_profile),
        )

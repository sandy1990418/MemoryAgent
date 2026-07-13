import ast
from pathlib import Path

import pytest

from memory_agent.application.structured_service import StructuredMemoryService
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.policies.event import EventMemoryPolicy
from memory_agent.policies.structured import (
    StructuredMemoryPolicy,
    get_memory_policy,
)
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def test_structured_and_event_policies_have_distinct_contract_names():
    assert EventMemoryPolicy is not StructuredMemoryPolicy


def test_removed_legacy_packages_and_modules_are_absent():
    removed = [
        Path("memory_agent/structured"),
        Path("memory_agent/profiles"),
        Path("memory_agent/evaluation"),
        Path("memory_agent/chat.py"),
        Path("memory_agent/application/memory_service.py"),
        Path("memory_agent/models/memory.py"),
        Path("memory_agent/models/policy.py"),
        Path("memory_agent/models/sections.py"),
        Path("memory_agent/models/transcript.py"),
    ]

    assert [str(path) for path in removed if path.exists()] == []


def test_structured_service_rejects_policy_mismatches_at_assembly_boundary():
    chat_policy = get_memory_policy("chat")
    agent_policy = get_memory_policy("agent")
    memory = Memory(CHAT_SECTIONS, policy=chat_policy)
    updater = MemoryUpdater(
        ScriptedLLM(lambda *_: "[]"),
        CHAT_SECTIONS,
        policy=agent_policy,
    )

    with pytest.raises(ValueError, match="conflicting policies"):
        StructuredMemoryService(
            memory=memory,
            updater=updater,
            policy=chat_policy,
        )


def test_canonical_packages_do_not_import_legacy_implementation_paths():
    roots = [
        Path("memory_agent/core"),
        Path("memory_agent/policies"),
        Path("memory_agent/normalization"),
        Path("memory_agent/update"),
        Path("memory_agent/retrieval"),
        Path("memory_agent/application"),
        Path("memory_agent/adapters"),
        Path("memory_agent/agents"),
    ]
    forbidden = (
        "memory_agent.structured",
        "memory_agent.profiles",
        "memory_agent.models.memory",
        "memory_agent.models.policy",
        "memory_agent.models.sections",
        "memory_agent.models.transcript",
        "memory_agent.chat",
    )
    violations = []
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = [
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            ]
            imported.extend(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            violations.extend(
                f"{path}:{name}"
                for name in imported
                if name.startswith(forbidden)
            )

    assert violations == []


def test_runtime_package_does_not_import_repo_evaluation_tooling():
    violations = []
    for path in Path("memory_agent").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        names.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        violations.extend(
            f"{path}:{name}"
            for name in names
            if name == "evaluation" or name.startswith("evaluation.")
        )

    assert violations == []


def test_core_has_no_policy_llm_framework_or_application_dependencies():
    forbidden = (
        "memory_agent.policies",
        "memory_agent.update",
        "memory_agent.retrieval",
        "memory_agent.application",
        "memory_agent.adapters",
        "memory_agent.clients",
        "langchain",
        "langgraph",
        "openai",
    )
    violations = []
    for path in Path("memory_agent/core").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        names.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        violations.extend(
            f"{path}:{name}" for name in names if name.startswith(forbidden)
        )

    assert violations == []

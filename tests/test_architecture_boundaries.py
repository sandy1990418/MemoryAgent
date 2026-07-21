"""Import and package-boundary contracts for the chat-only runtime."""

import ast
from pathlib import Path

from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.policies.structured import CHAT_POLICY


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    return names


def test_chat_is_the_only_production_policy_and_section_contract():
    assert CHAT_POLICY.name == "chat"
    assert CHAT_POLICY.max_ops_per_batch is None
    assert CHAT_POLICY.disallowed_sections == frozenset(
        {"exact_values", "timeline", "tool_facts"}
    )
    assert [section.key for section in CHAT_SECTIONS] == [
        "decisions",
        "preferences",
        "status_changes",
        "goal",
        "facts",
        "progress",
        "open_questions",
        "failed_attempts",
    ]


def test_removed_legacy_packages_and_modules_are_absent():
    removed = [
        Path("memory_agent/structured"),
        Path("memory_agent/profiles"),
        Path("memory_agent/evaluation"),
        Path("memory_agent/chat.py"),
        Path("memory_agent/agents"),
        Path("memory_agent/domain"),
        Path("memory_agent/longterm"),
        Path("memory_agent/clients/mem0.py"),
        Path("memory_agent/policies/event.py"),
        Path("memory_agent/application/memory_service.py"),
        Path("memory_agent/models/memory.py"),
        Path("memory_agent/models/policy.py"),
        Path("memory_agent/models/sections.py"),
        Path("memory_agent/models/transcript.py"),
    ]

    assert [str(path) for path in removed if path.exists()] == []


def test_chat_facade_has_no_reverse_dependency_on_optional_integrations():
    imported = _imports(Path("memory_agent/application/chat.py"))
    forbidden = (
        "memory_agent.agents",
        "memory_agent.clients.mem0",
        "memory_agent.domain",
        "memory_agent.longterm",
        "evaluation",
        "scripts",
        "langchain",
        "langgraph",
    )
    assert not any(name == item or name.startswith(item + ".") for name in imported for item in forbidden)


def test_runtime_package_does_not_import_evaluation_or_framework_integrations():
    roots = [
        Path("memory_agent/core"),
        Path("memory_agent/policies"),
        Path("memory_agent/normalization"),
        Path("memory_agent/update"),
        Path("memory_agent/retrieval"),
        Path("memory_agent/application"),
    ]
    forbidden = (
        "memory_agent.agents",
        "memory_agent.clients.mem0",
        "memory_agent.domain",
        "memory_agent.longterm",
        "memory_agent.adapters.langchain",
        "evaluation",
        "scripts",
        "langchain",
        "langgraph",
    )
    violations = []
    for root in roots:
        for path in root.rglob("*.py"):
            violations.extend(
                f"{path}:{name}"
                for name in _imports(path)
                if any(name == item or name.startswith(item + ".") for item in forbidden)
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
        violations.extend(
            f"{path}:{name}"
            for name in _imports(path)
            if name == forbidden or name.startswith(tuple(item + "." for item in forbidden))
        )

    assert violations == []

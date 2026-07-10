from pathlib import Path

from memory_agent.agents.hybrid import build_hybrid_agent
from memory_agent.agents.structured import build_structured_agent
from memory_agent.models.config import HybridAgentConfig, StructuredAgentConfig


def test_structured_agent_uses_only_injected_tools(monkeypatch):
    captured = {}
    tools = [object(), object()]

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("memory_agent.agents.structured.create_agent", fake_create_agent)

    build_structured_agent(StructuredAgentConfig(), tools=tools)

    assert captured["tools"] == tools


def test_hybrid_agent_uses_only_injected_tools(monkeypatch):
    captured = {}
    tools = [object()]

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("memory_agent.agents.hybrid.create_agent", fake_create_agent)

    build_hybrid_agent(
        HybridAgentConfig(mem0_backend="disabled"),
        tools=tools,
    )

    assert captured["tools"] == tools


def test_core_package_does_not_import_demo_modules():
    imports = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("memory_agent").rglob("*.py")
    )

    assert "from demos" not in imports
    assert "import demos" not in imports

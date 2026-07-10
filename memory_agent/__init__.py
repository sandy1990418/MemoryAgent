"""Structured conversational memory and LangChain-facing memory adapters.

The package root keeps backwards-compatible exports via lazy loading so
standalone chat imports do not eagerly pull agent, BEAM, or mem0 dependencies.
"""

from __future__ import annotations

_EXPORTS = {
    "LLMClient": ("memory_agent.clients.llm", "LLMClient"),
    "OpenAIClient": ("memory_agent.clients.llm", "OpenAIClient"),
    "TokenLedger": ("memory_agent.clients.llm", "TokenLedger"),
    "TokenUsage": ("memory_agent.clients.llm", "TokenUsage"),
    "LongTermMemory": ("memory_agent.clients.mem0", "LongTermMemory"),
    "Mem0LongTermMemory": ("memory_agent.clients.mem0", "Mem0LongTermMemory"),
    "HybridAgentConfig": ("memory_agent.models.config", "HybridAgentConfig"),
    "StructuredAgentConfig": ("memory_agent.models.config", "StructuredAgentConfig"),
    "ProductMemoryConfig": ("memory_agent.models.config", "ProductMemoryConfig"),
    "LongTermHit": ("memory_agent.models.longterm", "LongTermHit"),
    "MemoryEntry": ("memory_agent.models.memory", "MemoryEntry"),
    "SelectedMemory": ("memory_agent.models.memory", "SelectedMemory"),
    "MemoryPolicy": ("memory_agent.models.policy", "MemoryPolicy"),
    "get_memory_policy": ("memory_agent.models.policy", "get_memory_policy"),
    "StructuredAgentRuntime": ("memory_agent.models.runtime", "StructuredAgentRuntime"),
    "HybridAgentRuntime": ("memory_agent.models.runtime", "HybridAgentRuntime"),
    "AGENT_SECTIONS": ("memory_agent.models.sections", "AGENT_SECTIONS"),
    "CHAT_SECTIONS": ("memory_agent.models.sections", "CHAT_SECTIONS"),
    "EVAL_SECTIONS": ("memory_agent.models.sections", "EVAL_SECTIONS"),
    "PRACTICAL_SECTIONS": ("memory_agent.models.sections", "PRACTICAL_SECTIONS"),
    "SectionConfig": ("memory_agent.models.sections", "SectionConfig"),
    "MemoryCompactor": ("memory_agent.structured.compactor", "MemoryCompactor"),
    "Memory": ("memory_agent.structured.memory", "Memory"),
    "MemorySelector": ("memory_agent.structured.selector", "MemorySelector"),
    "MemorySession": ("memory_agent.structured.session", "MemorySession"),
    "Transcript": ("memory_agent.structured.transcript", "Transcript"),
    "MemoryUpdater": ("memory_agent.structured.updater", "MemoryUpdater"),
    "UpdateFailed": ("memory_agent.structured.updater", "UpdateFailed"),
    "Turn": ("memory_agent.models.transcript", "Turn"),
    "WorkingWindow": ("memory_agent.structured.window", "WorkingWindow"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

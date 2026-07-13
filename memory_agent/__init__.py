"""Structured conversational memory and LangChain-facing memory adapters.

The package root uses lazy public exports so standalone chat imports do not
eagerly pull agent or mem0 dependencies.
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
    "MemoryEntry": ("memory_agent.core.models", "MemoryEntry"),
    "SelectedMemory": ("memory_agent.core.models", "SelectedMemory"),
    "StructuredMemoryPolicy": ("memory_agent.policies.structured", "StructuredMemoryPolicy"),
    "get_memory_policy": ("memory_agent.policies.structured", "get_memory_policy"),
    "StructuredAgentRuntime": ("memory_agent.models.runtime", "StructuredAgentRuntime"),
    "HybridAgentRuntime": ("memory_agent.models.runtime", "HybridAgentRuntime"),
    "AGENT_SECTIONS": ("memory_agent.core.sections", "AGENT_SECTIONS"),
    "CHAT_SECTIONS": ("memory_agent.core.sections", "CHAT_SECTIONS"),
    "EVAL_SECTIONS": ("memory_agent.core.sections", "EVAL_SECTIONS"),
    "PRACTICAL_SECTIONS": ("memory_agent.core.sections", "PRACTICAL_SECTIONS"),
    "SectionConfig": ("memory_agent.core.sections", "SectionConfig"),
    "MemoryCompactor": ("memory_agent.update.compactor", "MemoryCompactor"),
    "Memory": ("memory_agent.core.store", "Memory"),
    "MemorySelector": ("memory_agent.retrieval.selector", "MemorySelector"),
    "MemorySession": ("memory_agent.application.session", "MemorySession"),
    "StructuredMemoryService": ("memory_agent.application.structured_service", "StructuredMemoryService"),
    "Transcript": ("memory_agent.core.transcript_store", "Transcript"),
    "MemoryUpdater": ("memory_agent.update.updater", "MemoryUpdater"),
    "UpdateFailed": ("memory_agent.update.updater", "UpdateFailed"),
    "Turn": ("memory_agent.core.transcript", "Turn"),
    "WorkingWindow": ("memory_agent.core.window", "WorkingWindow"),
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

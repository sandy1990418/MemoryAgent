from memory_agent.domain import EventSourceType, MemoryEntry, MemoryScope, MemoryType
from memory_agent.policies.event import EventMemoryCandidate


class AgentEventMemoryPolicy:
    name = "agent"

    def should_store(self, candidate: EventMemoryCandidate) -> bool:
        if candidate.event.source_type == EventSourceType.TOOL_RESULT:
            return bool(candidate.event.metadata.get("durable_summary"))
        return candidate.event.source_type in {EventSourceType.AGENT_DECISION, EventSourceType.TASK_STATUS, EventSourceType.OBSERVATION, EventSourceType.ARTIFACT}

    def classify(self, candidate: EventMemoryCandidate) -> MemoryType:
        if candidate.suggested_type:
            return candidate.suggested_type
        source = candidate.event.source_type
        if source == EventSourceType.AGENT_DECISION:
            return MemoryType.DECISION
        if source == EventSourceType.TASK_STATUS:
            text = candidate.content.lower()
            return MemoryType.UNRESOLVED_ISSUE if "block" in text or "fail" in text else MemoryType.TASK_STATE
        if source == EventSourceType.ARTIFACT:
            return MemoryType.ARTIFACT_REFERENCE
        return MemoryType.OBSERVATION

    def importance(self, candidate: EventMemoryCandidate) -> float:
        return .95 if self.classify(candidate) in {MemoryType.TASK_STATE, MemoryType.UNRESOLVED_ISSUE, MemoryType.FAILED_ATTEMPT} else .75

    def retention_priority(self, entry: MemoryEntry) -> float:
        weights = {MemoryType.GOAL: 1.0, MemoryType.UNRESOLVED_ISSUE: .98, MemoryType.TASK_STATE: .95, MemoryType.DECISION: .9, MemoryType.FAILED_ATTEMPT: .85, MemoryType.ARTIFACT_REFERENCE: .8}
        return weights.get(entry.memory_type, .5) * entry.importance

    def scope_for(self, candidate: EventMemoryCandidate) -> MemoryScope:
        return MemoryScope.TASK

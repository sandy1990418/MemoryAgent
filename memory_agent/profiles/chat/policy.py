from memory_agent.domain import MemoryEntry, MemoryScope, MemoryType
from memory_agent.profiles.policy import MemoryCandidate


class ChatMemoryPolicy:
    name = "chat"
    _durable_markers = ("prefer", "always", "never", "my ", "we decided", "goal", "blocked")

    def should_store(self, candidate: MemoryCandidate) -> bool:
        text = candidate.content.strip().lower()
        if candidate.event.actor == "assistant" and not candidate.event.metadata.get("accepted"):
            return False
        return bool(candidate.suggested_type or any(marker in text for marker in self._durable_markers))

    def classify(self, candidate: MemoryCandidate) -> MemoryType:
        if candidate.suggested_type:
            return candidate.suggested_type
        text = candidate.content.lower()
        if "prefer" in text:
            return MemoryType.USER_PREFERENCE
        if "goal" in text:
            return MemoryType.GOAL
        if "blocked" in text:
            return MemoryType.UNRESOLVED_ISSUE
        return MemoryType.USER_FACT

    def importance(self, candidate: MemoryCandidate) -> float:
        return 0.9 if self.classify(candidate) in {MemoryType.USER_PREFERENCE, MemoryType.USER_CONSTRAINT, MemoryType.GOAL} else 0.65

    def retention_priority(self, entry: MemoryEntry) -> float:
        weights = {MemoryType.USER_FACT: 1.0, MemoryType.USER_PREFERENCE: .95, MemoryType.USER_CONSTRAINT: .9, MemoryType.GOAL: .85, MemoryType.TASK_STATE: .6}
        return weights.get(entry.memory_type, .4) * entry.importance

    def scope_for(self, candidate: MemoryCandidate) -> MemoryScope:
        return MemoryScope.USER if self.classify(candidate) in {MemoryType.USER_FACT, MemoryType.USER_PREFERENCE, MemoryType.USER_CONSTRAINT} else MemoryScope.SESSION

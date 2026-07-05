from collections.abc import Callable


class ScriptedLLM:
    """Deterministic fake LLM for tests: delegates to a callback."""

    def __init__(self, script: Callable[[str, list[dict]], str]) -> None:
        self.script = script

    def complete(self, system: str, messages: list[dict], model: str | None = None) -> str:
        return self.script(system, messages)

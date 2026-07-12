"""Chat-language subject normalization for deterministic value updates."""

from __future__ import annotations

import re

from memory_agent.models.memory import MemoryValue, SubjectIdentity


_VALUE_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?:is|was|reached|became|(?:has\s+)?(?:changed|improved|increased|decreased|reduced)(?:\s+to)?)\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>%|percent|percentage|ms|milliseconds?|s|seconds?|[A-Za-z]+)?\.?$",
    re.IGNORECASE,
)
_QUALIFIER_RE = re.compile(r"\b(?P<qualifier>(?:if|when|while|unless)\s+.+?)(?:,|\s+then\b)", re.IGNORECASE)
_LEADING_OWNER_RE = re.compile(r"^(?:the|my|our|a|an)\s+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_UNITS = {
    "%": "percent",
    "percentage": "percent",
    "milliseconds": "ms",
    "millisecond": "ms",
    "seconds": "s",
    "second": "s",
}
_PERSONAL_CUE_RE = re.compile(
    r"(?P<subject>(?:[A-Za-z][\w'-]*\s+){0,4}(?P<attribute>budget|goal|target|rate|duration))",
    re.IGNORECASE,
)
_TAKES_RE = re.compile(
    r"(?P<subject>(?:[A-Za-z][\w'-]*\s+){1,5})(?:now\s+)?takes?\s+"
    r"(?P<value>\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*"
    r"(?P<unit>days?|weeks?|months?|years?|hours?|minutes?)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?P<currency>[$€£])?\s*(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|books?|pages?|days?|weeks?|months?|years?|hours?|minutes?)?",
    re.IGNORECASE,
)


class ChatSubjectNormalizer:
    """Conservative linguistic normalizer owned by the chat profile.

    The full subject phrase is retained as the entity. This deliberately favors
    false negatives over merging two similarly-worded but distinct subjects.
    """

    namespace = "chat"

    def normalize(self, text: str) -> tuple[SubjectIdentity, MemoryValue] | None:
        compact = _SPACE_RE.sub(" ", text.strip())
        qualifier_match = _QUALIFIER_RE.search(compact)
        qualifier = self._canonical(qualifier_match.group("qualifier")) if qualifier_match else None
        if qualifier_match:
            compact = (compact[: qualifier_match.start()] + compact[qualifier_match.end() :]).strip(" ,")
        match = _VALUE_RE.match(compact)
        if not match:
            return self._normalize_personal_value(compact)
        subject = self._canonical(_LEADING_OWNER_RE.sub("", match.group("subject")))
        if len(subject.split()) < 2:
            return None
        raw_unit = (match.group("unit") or "").lower()
        unit = _UNITS.get(raw_unit, raw_unit or None)
        return (
            SubjectIdentity(
                namespace=self.namespace,
                entity=subject,
                attribute="value",
                qualifier=qualifier,
                confidence=0.9,
            ),
            MemoryValue(value=match.group("value"), unit=unit),
        )

    def _normalize_personal_value(
        self, compact: str
    ) -> tuple[SubjectIdentity, MemoryValue] | None:
        compact = re.sub(r"^User stated:\s*", "", compact, flags=re.IGNORECASE)
        takes = _TAKES_RE.search(compact)
        if takes:
            return (
                SubjectIdentity(
                    self.namespace,
                    self._canonical(takes.group("subject")),
                    "duration",
                    confidence=0.9,
                ),
                MemoryValue(
                    _SPACE_RE.sub("", takes.group("value")),
                    takes.group("unit").lower().rstrip("s"),
                ),
            )
        cue = _PERSONAL_CUE_RE.search(compact)
        if cue is None:
            return None
        numbers = list(_NUMBER_RE.finditer(compact))
        if not numbers:
            return None
        # Prefer the closest value to the semantic cue; this selects $35 for
        # "spent $43, over my $35 monthly budget" rather than the spend.
        number = min(
            numbers,
            key=lambda match: min(
                abs(match.start() - cue.end()), abs(match.end() - cue.start())
            ),
        )
        currency = number.group("currency")
        raw_unit = (number.group("unit") or "").lower()
        unit = currency or _UNITS.get(raw_unit, raw_unit.rstrip("s") or None)
        return (
            SubjectIdentity(
                namespace=self.namespace,
                entity=self._canonical(cue.group("subject")),
                attribute=cue.group("attribute").lower(),
                confidence=0.9,
            ),
            MemoryValue(value=number.group("value").replace(",", ""), unit=unit),
        )

    @staticmethod
    def _canonical(value: str) -> str:
        return _SPACE_RE.sub(" ", value.strip().lower().rstrip("."))

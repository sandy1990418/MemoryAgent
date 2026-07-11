"""Bounded, deterministic related-memory selection for update prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from memory_agent.models.memory import MemoryEntry, SubjectNormalizer
from memory_agent.models.transcript import Turn
from memory_agent.structured.heuristics import content_words
from memory_agent.structured.memory import Memory


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class UpdateMemoryMatch:
    entry: MemoryEntry
    score: float
    reasons: tuple[str, ...]
    score_components: tuple[tuple[str, float], ...]
    confidence: float


@dataclass(frozen=True)
class UpdateMemorySelection:
    matches: tuple[UpdateMemoryMatch, ...]
    visible_tokens: int
    fallback_used: bool = False
    fallback_reason: str | None = None

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(match.entry for match in self.matches)


class UpdateMemorySelector:
    """Select only entries lexically related to turns, without domain identities."""

    def __init__(
        self,
        memory: Memory,
        token_estimator: Callable[[str], int] | None = None,
        subject_normalizer: SubjectNormalizer | None = None,
        identity_confidence_threshold: float = 0.85,
        max_legacy_fallback_entries: int = 4,
    ) -> None:
        self.memory = memory
        self.token_estimator = token_estimator or _default_token_estimator
        self.subject_normalizer = subject_normalizer
        self.identity_confidence_threshold = identity_confidence_threshold
        self.max_legacy_fallback_entries = max_legacy_fallback_entries

    @staticmethod
    def _typed_key(normalized: tuple | None) -> tuple[str, str, str, str | None, str | None] | None:
        if normalized is None:
            return None
        identity, value = normalized
        return (identity.namespace, identity.entity, identity.attribute, identity.qualifier, value.unit)

    def select_for_update(
        self,
        turns: list[Turn],
        budget: int | None,
    ) -> UpdateMemorySelection:
        query_words = content_words(
            "\n".join(
                turn.content for turn in turns if turn.role in {"user", "assistant"}
            )
        )
        if not query_words or budget == 0:
            return UpdateMemorySelection(matches=(), visible_tokens=0)

        turn_identities = set()
        if self.subject_normalizer is not None:
            for turn in turns:
                normalized = self.subject_normalizer.normalize(turn.content)
                if normalized is not None and normalized[0].confidence >= self.identity_confidence_threshold:
                    turn_identities.add(self._typed_key(normalized))

        typed_candidates: list[UpdateMemoryMatch] = []
        legacy_candidates: list[UpdateMemoryMatch] = []
        for entry in self.memory.entries.values():
            entry_key = None
            if entry.subject_identity is not None and entry.value is not None:
                entry_key = self._typed_key((entry.subject_identity, entry.value))
            elif self.subject_normalizer is not None:
                normalized = self.subject_normalizer.normalize(entry.text)
                if normalized is not None and normalized[0].confidence >= self.identity_confidence_threshold:
                    entry_key = self._typed_key(normalized)
            if entry_key is not None and entry_key in turn_identities:
                typed_candidates.append(UpdateMemoryMatch(
                    entry=entry, score=100.0, reasons=("typed_exact_subject_unit_qualifier",),
                    score_components=(("typed_exact", 100.0),), confidence=1.0,
                ))
                continue
            entry_words = content_words(entry.text)
            overlap = query_words & entry_words
            if not overlap:
                continue
            lexical = float(len(overlap) * 3)
            active = 2.0 if entry.status == "active" else 0.0
            recency = (
                min(max(entry.provenance), 1000) / 1000.0 if entry.provenance else 0.0
            )
            score = lexical + active + recency
            coverage = len(overlap) / max(1, min(len(query_words), len(entry_words)))
            confidence = min(1.0, coverage)
            reasons = (f"lexical_overlap:{len(overlap)}",) + (
                ("active",) if active else ("superseded",)
            )
            legacy_candidates.append(
                UpdateMemoryMatch(
                    entry=entry,
                    score=score,
                    reasons=reasons,
                    score_components=(
                        ("lexical_overlap", lexical),
                        ("active", active),
                        ("recency", recency),
                    ),
                    confidence=confidence,
                )
            )
        typed_candidates.sort(key=lambda match: match.entry.id)
        legacy_candidates.sort(key=lambda match: (-match.score, match.entry.id))
        # Lexical matching is a bounded compatibility lane. It is used only when
        # exact typed identity could not fully identify the related legacy state.
        fallback_used = bool(legacy_candidates)
        candidates = typed_candidates + legacy_candidates[: self.max_legacy_fallback_entries]

        selected: list[UpdateMemoryMatch] = []
        visible_tokens = 0
        for candidate in candidates:
            rendered = self.memory.render(
                include_superseded=True,
                entries=[candidate.entry],
            )
            tokens = self.token_estimator(rendered)
            if budget is not None and visible_tokens + tokens > budget:
                continue
            selected.append(candidate)
            visible_tokens += tokens
        return UpdateMemorySelection(
            tuple(selected), visible_tokens, fallback_used,
            "bounded_ambiguous_legacy_lexical_match" if fallback_used else None,
        )

"""Prompt construction for structured-memory update operations."""

from __future__ import annotations

import json
import re

from memory_agent.core.sections import SectionConfig
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import StructuredMemoryPolicy


_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_MAX_UPDATER_TURN_CHARS = 4000
_MAX_UPDATER_BATCH_CHARS = 8000


def _compact_turn_content(content: str) -> str:
    """Remove re-derivable code payloads and bound pathological chat turns."""
    compact = _CODE_BLOCK_RE.sub("[code block omitted from memory extraction]", content)
    if len(compact) <= _MAX_UPDATER_TURN_CHARS:
        return compact
    head = _MAX_UPDATER_TURN_CHARS * 2 // 3
    tail = _MAX_UPDATER_TURN_CHARS - head
    return (
        compact[:head]
        + "\n[long turn middle omitted from LLM extraction]\n"
        + compact[-tail:]
    )


def _compact_turns_for_prompt(turns: list[Turn]) -> list[dict]:
    per_turn = max(500, _MAX_UPDATER_BATCH_CHARS // max(1, len(turns)))
    compacted = []
    for turn in turns:
        content = _compact_turn_content(turn.content)
        if len(content) > per_turn:
            head = per_turn * 2 // 3
            tail = per_turn - head
            content = content[:head] + "\n[turn shortened]\n" + content[-tail:]
        compacted.append({"turn_id": turn.id, "role": turn.role, "content": content})
    return compacted


def _chat_updater_system(
    *,
    sections_block: str,
    current_memory: str,
    turns_block: str,
) -> str:
    return (
        "Maintain sparse structured chat memory. Return a JSON array only.\n\n"
        f"Sections:\n{sections_block}\n\n"
        "Operations:\n"
        '- ADD {"op":"ADD","section":<key>,"text":<text>,"provenance":[turn ids]}\n'
        '- UPDATE {"op":"UPDATE","id":<exact memory id>,"text":<text>,"provenance":[turn ids]}\n'
        '- SUPERSEDE {"op":"SUPERSEDE","id":<exact memory id>,"reason":<reason>}\n'
        '- NOOP {"op":"NOOP"}\n\n'
        "CHAT POLICY rules:\n"
        "- Default to NOOP. Save durable user preferences, instructions, decisions, goals, current project state, observed results, blockers, failed attempts, and explicit corrections.\n"
        "- For a substantive completed user/assistant exchange, ADD at most one topic-scoped progress entry summarizing the concrete methods, comparisons, outcomes, or learning progression covered across both turns. Use neutral wording such as 'Discussion covered'; never claim the user implemented or accepted assistant content.\n"
        "- Do not save a bare question, greeting, short answer, generic recommendation, or unaccepted proposal as progress. Do not copy snippets turn-by-turn.\n"
        "- Do not save generic assistant advice as a user or project fact.\n"
        "- Keep semantic entries normally under 25 words. A progress entry may use up to 60 words when needed to preserve concrete chronology for later compaction. Prefer one consolidated entry per topic per batch. Do not infer missing details.\n"
        "- A batch may produce at most three concise ADD or UPDATE operations.\n"
        "- For a reversal, MUST SUPERSEDE the old active entry, then ADD a new replacement entry. Never use UPDATE for that case.\n"
        "- UPDATE/SUPERSEDE ids must appear in Current memory. Never use a turn_id as an entry id; if no exact id exists, ADD or NOOP.\n"
        "- Provenance must contain only turn ids from Turns JSON. Embed important versions, dates, counts, and metrics in their subject entry; do not create a value inventory.\n\n"
        f"Current memory:\n{current_memory}\n\n"
        f"Turns JSON:\n{turns_block}\n"
    )


def build_updater_prompt(
    *,
    sections: list[SectionConfig],
    policy: StructuredMemoryPolicy,
    current_memory: str,
    turns: list[Turn],
) -> tuple[str, list[dict]]:
    """Build the single chat updater contract.

    ``policy`` remains an explicit argument so callers can pass the shared
    ``CHAT_POLICY`` object, but it is not a profile switch.  Evaluation labels
    and runner-specific rubrics must never alter this production prompt.
    """
    sections_block = "\n".join(
        f'- key="{section.key}" prefix="{section.prefix}": {section.description}'
        for section in sections
    )
    turns_block = json.dumps(
        _compact_turns_for_prompt(turns),
        ensure_ascii=False,
        indent=2,
    )
    system = _chat_updater_system(
        sections_block=sections_block,
        current_memory=current_memory,
        turns_block=turns_block,
    )
    return system, [{
        "role": "user",
        "content": "Apply the rules above and return the ops JSON array for these turns.",
    }]

def build_compactor_prompt(
    *,
    sections: list[SectionConfig],
    current_memory: str,
) -> tuple[str, list[dict]]:
    sections_block = "\n".join(
        f'- key="{section.key}": {section.description}' for section in sections
    )
    system = (
        "Compact structured memory by semantic subject, not as a transcript summary. "
        "Return a JSON array containing only SUPERSEDE, ADD, or NOOP.\n\n"
        "Available sections:\n"
        f"{sections_block}\n\n"
        "Rules:\n"
        "1. Merge only active entries about the same semantic subject.\n"
        "2. SUPERSEDE every replaced active entry, then ADD one canonical entry.\n"
        "3. Preserve latest truth, preferences, decisions, current state, blockers, "
        "and failed attempts.\n"
        "3a. A progress rollup is a topic summary, not a latest-value overwrite.\n"
        "4. Canonical ADD provenance must be the union of source provenance ids.\n"
        "5. Never supersede without a canonical ADD; never delete entries.\n"
        "6. Never operate on or re-activate a superseded entry.\n"
        "7. Do not merge entries only because they share broad words.\n"
        "8. Use NOOP when no safe same-subject consolidation exists.\n"
        "9. Return the JSON array only.\n\n"
        "Current memory:\n"
        f"{current_memory}"
    )
    return system, [
        {
            "role": "user",
            "content": "Return subject-based compaction operations for the active entries.",
        }
    ]


def build_progress_rollup_prompt(
    *,
    source_entries: str,
    max_chars: int,
) -> tuple[str, list[dict]]:
    """Prompt for content generation only; the application owns all memory ops."""
    system = (
        "Summarize the supplied progress entries into one compact chronological "
        "topic summary. Preserve concrete methods, outcomes, decisions, and important "
        "distinctions, while removing repetition. Do not mention memory ids, source "
        "entries, provenance, compaction, or superseding. Return plain summary text "
        f"only, with no JSON, markdown, label, or preamble. Maximum {max_chars} "
        "characters.\n\nProgress entries:\n"
        f"{source_entries}"
    )
    return system, [{"role": "user", "content": "Return the compact summary only."}]

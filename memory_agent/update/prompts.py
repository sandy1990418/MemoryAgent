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
        "- Default to NOOP for conversational noise. Save explicit durable user self-reports, including positive or negative reports about their experience, completed or ongoing work, current project state, preferences, goals, decisions, blockers, measured results, failed attempts, and corrections.\n"
        "- Evidence hierarchy: direct user assertions, corrections, explicit decisions, and reported outcomes are primary evidence of user or project state. Assistant messages are context only; suggestions, examples, plans, generated implementation detail, and claims about what happened are not user or project facts unless the user explicitly confirms or reports the outcome.\n"
        "- When direct user assertions conflict and neither assertion explicitly corrects or replaces the other, use Status Changes (key status_changes) for a concise unresolved-uncertainty entry that preserves both claims. Keep it active; do not choose a winner, UPDATE, or SUPERSEDE either claim.\n"
        "- An explicit user correction or replacement establishes the latest truth: MUST SUPERSEDE the old active entry, then ADD the corrected entry. Do not treat an assistant correction or suggestion as a user correction.\n"
        "- Preserve direct durable user state (preferences, goals, facts, corrections, and decisions) alongside useful discussion/progress rollups; never present assistant-derived progress as user-confirmed state.\n"
        "- Retain explicit user-stated goals as active across topic changes in Task Goal (key goal), never absorb them into Progress. UPDATE a goal when later user turns report progress toward that same goal; SUPERSEDE it only when the user explicitly completes, cancels, or replaces it. Assistant-proposed goals are not user goals unless the user explicitly accepts them.\n"
        "- Progress may retain either a direct user-reported milestone or one future-useful topic summary from a substantive completed exchange. Label the distinction in the text: use 'User reported:' for user-confirmed work/results and 'Discussion covered (implementation not confirmed):' for neutral topic coverage. Never present assistant explanations, examples, plans, generated code, or proposed outcomes as completed user work.\n"
        "- Do not save a bare question, greeting, short answer, generic recommendation, or unaccepted proposal as progress. Do not copy snippets turn-by-turn.\n"
        "- Do not save generic assistant advice as a user or project fact.\n"
        "- Keep semantic entries normally under 25 words. A progress entry may use up to 60 words when needed to preserve concrete chronology for later compaction. Prefer one consolidated entry per topic per batch. Do not infer missing details.\n"
        "- Before ADD, inspect the active entries for the same semantic subject. If the turns are already represented, use NOOP; if they add material detail to that subject, UPDATE the exact active entry and consolidate the latest text instead of creating another entry.\n"
        "- Never create a second active entry merely because equivalent information uses different wording. When the subject match is uncertain or no exact visible id is available, prefer NOOP over a near-duplicate ADD. Keep genuinely distinct claims separate.\n"
        "- Do not omit a distinct durable assertion merely to keep a batch artificially small. Keep every operation concise, preserve its provenance, and avoid near-duplicates.\n"
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
    ``CHAT_POLICY`` object. Evaluation labels and runner-specific rubrics must
    never alter this production prompt.
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
        "Output schema (use these keys exactly; do not invent aliases):\n"
        '- SUPERSEDE: {"op":"SUPERSEDE","id":"F1","reason":"same subject merged"}\n'
        '- ADD: {"op":"ADD","section":"facts","text":"canonical entry text","provenance":[1,2]}\n'
        '- NOOP: {"op":"NOOP"}\n'
        "SUPERSEDE takes only an existing entry id and reason. ADD takes only a valid section key, non-empty text, and provenance turn ids; it does not take id, key, entry, or value fields. Do not emit UPDATE during compaction.\n\n"
        "Rules:\n"
        "1. Merge only active entries about the same semantic subject.\n"
        "2. SUPERSEDE every replaced active entry, then ADD one canonical entry.\n"
        "3. Preserve latest truth, preferences, decisions, current state, blockers, "
        "and failed attempts.\n"
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

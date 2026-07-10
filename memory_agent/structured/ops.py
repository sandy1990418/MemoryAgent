"""Shared parsing contract for updater and compactor operation arrays."""

from __future__ import annotations

import json
import re


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class UpdateFailed(Exception):
    """Raised when an LLM response cannot be converted into memory operations."""


def parse_memory_ops(response: str) -> list[dict] | None:
    text = response.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    for match in _CODE_FENCE_RE.finditer(text):
        for candidate in (match.group(0), match.group(1).strip()):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                return parsed

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


"""Frozen, content-addressed memory-evaluation manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


def content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_state(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, text=True, capture_output=True,
        ).stdout

    commit = git("rev-parse", "HEAD").strip()
    # Generated evidence is derived from source and must not recursively change
    # the source digest on each regeneration.
    diff = git("diff", "--binary", "HEAD", "--", ".", ":(exclude)evaluation/artifacts")
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    untracked = [name for name in untracked if not name.startswith("evaluation/artifacts/")]
    untracked_payload = []
    for name in sorted(untracked):
        path = repo / name
        untracked_payload.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    dirty_payload = {"diff": diff, "untracked": untracked_payload}
    return {
        "commit": commit,
        "dirty": bool(diff or untracked),
        "dirty_digest": content_hash(dirty_payload),
    }


def _freeze_records(records: Sequence[Mapping[str, Any]], id_key: str) -> list[dict[str, str]]:
    frozen = []
    for record in records:
        if id_key not in record:
            raise ValueError(f"record is missing {id_key}")
        frozen.append({"id": str(record[id_key]), "content_hash": content_hash(record)})
    return sorted(frozen, key=lambda item: item["id"])


def build_frozen_manifest(
    *, repo: Path, resolved_configs: Mapping[str, Any], resolved_models: Mapping[str, Any],
    dataset: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]], development_case_ids: Sequence[str],
    holdout_case_ids: Sequence[str], route: str, token_count_provenance: Mapping[str, Any],
    historical_artifact: Mapping[str, Any] | None = None,
    historical_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    case_records = _freeze_records(cases, "id")
    probe_records = _freeze_records(probes, "id")
    known = {item["id"] for item in case_records}
    development = sorted(map(str, development_case_ids))
    holdout = sorted(map(str, holdout_case_ids))
    if set(development) & set(holdout):
        raise ValueError("development and holdout case IDs must be disjoint")
    if not set(development + holdout) <= known:
        raise ValueError("split contains an unknown case ID")
    if historical_artifact is None and not (historical_unavailable_reason or "").strip():
        raise ValueError("missing historical artifact requires an unavailable reason")
    historical = deepcopy(historical_artifact) if historical_artifact is not None else {
        "status": "unavailable", "reason": historical_unavailable_reason,
    }
    if historical_artifact is not None and "routing_provenance" not in historical:
        raise ValueError("historical artifact requires routing_provenance")
    return {
        "schema_version": "production-memory-manifest/v1",
        "source": source_state(repo),
        "resolved_configs": deepcopy(dict(resolved_configs)),
        "resolved_models": deepcopy(dict(resolved_models)),
        "dataset": {"id": str(dataset["id"]), "content_hash": content_hash(dataset)},
        "cases": case_records, "probes": probe_records,
        "splits": {
            "development": {"case_ids": development, "content_hash": content_hash(development)},
            "holdout": {"case_ids": holdout, "content_hash": content_hash(holdout)},
        },
        "route": route, "token_count_provenance": deepcopy(dict(token_count_provenance)),
        "historical_artifact": historical,
    }

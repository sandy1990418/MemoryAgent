from pathlib import Path

import pytest

from memory_agent.evaluation.final_report import (
    build_final_report, build_paired_routing_result, validate_final_report,
)
from memory_agent.evaluation.manifest import build_frozen_manifest


def test_manifest_freezes_source_content_splits_and_historical_provenance():
    manifest = build_frozen_manifest(
        repo=Path("."), resolved_configs={"answer_tokens": 500},
        resolved_models={"answer": "model-a", "judge": "model-j"},
        dataset={"id": "beam-100k", "version": 1},
        cases=[{"id": "1", "turns": ["a"]}, {"id": "2", "turns": ["b"]}],
        probes=[{"id": "p1", "question": "q"}], development_case_ids=["1"],
        holdout_case_ids=["2"], route="production",
        token_count_provenance={"policy": "characters_divided_by_four"},
        historical_artifact={"path": "old.json", "routing_provenance": "oracle-like"},
    )
    assert len(manifest["source"]["dirty_digest"]) == 64
    assert manifest["splits"]["development"]["case_ids"] == ["1"]
    assert manifest["historical_artifact"]["routing_provenance"] == "oracle-like"
    assert all(len(item["content_hash"]) == 64 for item in manifest["cases"] + manifest["probes"])


def test_manifest_requires_historical_unavailable_reason_and_disjoint_split():
    common = dict(repo=Path("."), resolved_configs={}, resolved_models={}, dataset={"id": "d"},
                  cases=[{"id": "1"}], probes=[{"id": "p"}], route="production",
                  token_count_provenance={})
    with pytest.raises(ValueError, match="unavailable reason"):
        build_frozen_manifest(**common, development_case_ids=["1"], holdout_case_ids=[])
    with pytest.raises(ValueError, match="disjoint"):
        build_frozen_manifest(**common, development_case_ids=["1"], holdout_case_ids=["1"],
                              historical_unavailable_reason="not retained")


def test_paired_contract_freezes_tolerance_gap_and_ability_status():
    result = build_paired_routing_result(
        production_score=.70, oracle_score=.75, paired_rubric_denominator=100,
        sample_size=20, abilities={"recall": {"production_score": .8, "oracle_score": .7}},
    )
    assert result["production_tolerance"] == .02
    assert result["gap"] == pytest.approx(.05)
    assert result["abilities"]["recall"]["status"] == "improved"
    assert result["production_passed"] is False


def test_provider_failure_is_validation_gap_and_oracle_cannot_offset():
    result = build_paired_routing_result(
        production_score=None, oracle_score=.9, paired_rubric_denominator=10, sample_size=3,
        abilities={}, provider_failures=[{"route": "production", "reason": "rate limit"}],
    )
    assert result["validation_gap"] is True
    assert result["gap"] is None and result["production_passed"] is None
    assert result["oracle_cannot_offset_production"] is True
    report = build_final_report(candidate={"routing": result})
    validate_final_report(report)


def test_final_report_rejects_available_section_without_mandatory_inner_fields():
    report = build_final_report()
    report["candidate"]["quality"] = {"canonical": 1}
    with pytest.raises(ValueError, match="candidate.quality missing"):
        validate_final_report(report)

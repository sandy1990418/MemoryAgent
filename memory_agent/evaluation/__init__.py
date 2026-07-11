"""Evaluation interfaces; datasets live outside the production domain."""
from memory_agent.evaluation.final_report import (
    build_final_report, build_paired_routing_result, unavailable, validate_final_report,
)
from memory_agent.evaluation.manifest import build_frozen_manifest, content_hash
from memory_agent.evaluation.online_simulation import (
    OnlineSimulation,
    SimulationMode,
    TranscriptExchange,
)

__all__ = [
    "OnlineSimulation", "SimulationMode", "TranscriptExchange",
    "build_final_report", "build_paired_routing_result", "unavailable", "validate_final_report",
    "build_frozen_manifest", "content_hash",
]

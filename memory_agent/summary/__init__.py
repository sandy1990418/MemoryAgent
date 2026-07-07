"""Summary-related agent code.

This package is only for the baseline LangChain `SummarizationMiddleware`
path. Structured memory is not summary-based and lives in
`memory_agent.structured`.
"""

from memory_agent.summary.agent import build_summary_agent

__all__ = ["build_summary_agent"]

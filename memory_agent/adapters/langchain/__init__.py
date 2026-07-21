"""Optional LangChain adapters.

Importing this package requires ``langchain-core``.  The canonical structured
chat adapter is intentionally separate from LangChain's agent middleware and
does not import LangGraph or any agent runtime.
"""

from .chat import LangChainChatAdapter

__all__ = [
    "LangChainChatAdapter",
]

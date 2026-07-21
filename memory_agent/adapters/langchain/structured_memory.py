"""Migration import for the chat-oriented LangChain adapter.

Structured memory is no longer an agent middleware.  Keep this import path
small and dependency-neutral for applications migrating from the pre-chat-only
surface; the implementation and public contract live in :mod:`.chat`.
"""

from .chat import LangChainChatAdapter, _content_to_text  # noqa: F401

StructuredMemoryMiddleware = LangChainChatAdapter

__all__ = ["StructuredMemoryMiddleware"]

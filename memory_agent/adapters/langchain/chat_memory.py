"""Public import path for the optional LangChain chat adapter.

The implementation lives in :mod:`memory_agent.adapters.langchain.chat`; this
module gives the adapter an explicit, discoverable ``chat_memory`` path while
keeping one implementation and one state machine.
"""

from .chat import LangChainChatAdapter

__all__ = [
    "LangChainChatAdapter",
]

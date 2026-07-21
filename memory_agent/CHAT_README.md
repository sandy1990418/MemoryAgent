# Chat memory

`memory_agent.application.chat` is the canonical standalone chat-memory
facade. It uses the single `CHAT_POLICY` and `CHAT_SECTIONS` contract while
keeping optional framework integrations outside the import path.

```python
from memory_agent.application.chat import build_chat_memory
from memory_agent.core.transcript import Turn

chat = build_chat_memory()  # Reads configs/product.yaml and environment.
chat.update([
    Turn(id=1, role="user", content="Remember I prefer concise replies."),
])
print(chat.render())
print(chat.token_usage())
```

`build_chat_memory` accepts an injected `LLMClient` for deterministic tests,
and `compact=False` disables optional active-entry compaction. Configuration
controls model and token limits while retention remains chat-only.

The facade returns applied and rejected operations from `ChatMemory.update`.
Source turns should remain available to the caller when an update is rejected,
so a later call can retry without losing context.

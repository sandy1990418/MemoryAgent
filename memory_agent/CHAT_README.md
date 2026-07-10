# Chat practical memory

`memory_agent.chat` is the standalone practical-memory facade. It avoids agent,
BEAM/evaluation, and mem0 imports so another engineer can copy or depend on the
chat memory surface without understanding the evaluation runners.

```python
from memory_agent.chat import build_chat_memory
from memory_agent.models.transcript import Turn

chat_memory = build_chat_memory()  # Reads configs/product.yaml, then env overrides.
chat_memory.update([Turn(id=1, role="user", content="Remember I prefer concise replies")])
print(chat_memory.render())
print(chat_memory.token_usage())  # Token spend per role: {"updater": {...}, "compactor": {...}}
```

Set `MEMORY_PROFILE`, `MEMORY_SECTIONS`, `MEMORY_COMPACTION_THRESHOLD`, or
`MEMORY_MODEL` to override one product setting without editing the YAML file.

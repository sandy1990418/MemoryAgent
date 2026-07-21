# MemoryAgent

MemoryAgent is a framework-free, token-bounded memory system for chat
applications. The supported product surface is intentionally small: one chat
policy, one section schema, and one application facade. Chat memory stores
durable user and project state while leaving ordinary questions and transient
assistant advice in the conversation window.

## Public API

```python
from memory_agent import Turn, build_chat_memory

chat = build_chat_memory()
chat.update([
    Turn(id=1, role="user", content="Remember that I prefer concise replies."),
])
print(chat.render())
print(chat.token_usage())
```

`memory_agent.application.chat.build_chat_memory` is the canonical builder.
It loads `configs/product.yaml`, applies supported environment overrides, and
does not import optional framework integrations or evaluation tooling.

## Project layout

```text
memory_agent/
  core/                    framework-neutral turns, entries, store, and schema
  policies/structured.py   the CHAT_POLICY contract
  normalization/           chat subject/value normalization
  update/                  extraction, validation, and compaction
  retrieval/               answer-time selection and rendering
  application/chat.py      canonical chat facade
  application/session.py   framework-free conversation session
  adapters/langchain/      optional LangChain chat adapter
  clients/llm.py           small LLM protocol and provider adapter
  models/config.py         product configuration

configs/product.yaml       chat memory limits and updater settings
demos/                     examples and local session helpers
tests/                     chat, storage, update, retrieval, and boundary tests
```

The runtime has no practical/agent/evaluation profile registry and no section
preset resolver. `CHAT_POLICY` and `CHAT_SECTIONS` are the sole production
contracts. Optional adapters are kept at the integration boundary and are not
required to import or use the plain chat facade.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` when using the default provider adapter. Tests use local
fake clients and do not require a network call.

Supported product settings include:

```text
MEMORY_MODEL
MEMORY_COMPACTION_THRESHOLD
UPDATE_MEMORY_TOKEN_BUDGET
EVICTED_TURN_TOKEN_BUDGET
UPDATER_MAX_CANDIDATE_ENTRIES
```

The updater applies `ADD`, `UPDATE`, `SUPERSEDE`, and `NOOP` operations
atomically. Invalid provenance, unsupported sections, oversized entries, or
partial batches are rejected without dropping source turns.

## Optional LangChain integration

Install the optional framework dependencies only when adapting an existing
LangChain chat loop:

```bash
pip install -r requirements-framework.txt
```

Use `memory_agent.adapters.langchain.chat_memory.LangChainChatAdapter` at that
boundary. The adapter delegates to the same chat facade and does not change
the production policy or section schema.

## Tests

```bash
python -m pytest -q
```

The architecture tests assert that the canonical chat runtime stays free of
optional integration imports and that removed surfaces do not return through
compatibility aliases.

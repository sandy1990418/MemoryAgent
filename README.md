# MemoryAgent

This repo has three runnable paths:

1. `react_summary_agent.py` is the recommended first version. It uses
   LangChain's built-in `SummarizationMiddleware` with a simple ReAct-style
   tool-calling agent.
2. `react_hybrid_agent.py` keeps `SummarizationMiddleware` for in-session
   compression and adds a mem0-backed long-term vector memory layer for
   cross-session recall.
3. `demo_react.py` is an experimental structured-memory version. It replaces
   `SummarizationMiddleware` with `StructuredMemoryMiddleware`, which evicts
   old messages into operation-based memory entries.

Start with the first path unless you specifically need cross-session semantic
recall, auditable memory entries, superseded history, or custom eviction
behavior.

For the full design, data flow, and memory policy details, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Primary Path: ReAct + SummarizationMiddleware

`react_summary_agent.py` keeps the architecture intentionally small:

- `create_agent(...)` owns the ReAct/tool-calling loop.
- `weather` and `calculator` are normal LangChain tools.
- `SummarizationMiddleware(...)` compresses older conversation context.
- `InMemorySaver()` keeps repeated turns in one LangGraph thread.

Run it with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
python react_summary_agent.py
```

Optional model overrides:

```bash
export MAIN_MODEL="openai:gpt-5.5"
export SUMMARY_MODEL="openai:gpt-5.4-mini"
```

The demo uses `trigger=("messages", 6)` and `keep=("messages", 2)` so
summarization happens quickly in a local test. For real use, prefer a token
trigger such as `trigger=("tokens", 4000)`.

## Hybrid Path: Summarization + Long-Term Vector Memory (mem0)

`react_hybrid_agent.py` adds a third runtime path for conversations that need
both short-term context compression and long-term recall. LangChain's
`SummarizationMiddleware` still handles most in-session context pressure. A
second middleware, placed after it, detects the message IDs that disappeared
from the active context, persists those turns to mem0, and injects semantically
relevant long-term memories into the system prompt on every model call.

Run it with:

```bash
export OPENAI_API_KEY="your-api-key"
python react_hybrid_agent.py
```

Optional knobs:

```bash
export MAIN_MODEL="openai:gpt-5.5"
export SUMMARY_MODEL="openai:gpt-5.4-mini"
export MEM0_USER_ID="demo-user"
export MEM0_DATA_DIR=".mem0"
```

The local mem0 store persists under `.mem0/`, including embedded Qdrant data
and mem0's history database. Re-running the script with the same
`MEM0_USER_ID` and `MEM0_DATA_DIR` demonstrates cross-session recall. Embedded
Qdrant takes an exclusive file lock on its local data path, so run only one
process against the same `.mem0/` directory at a time.

If `mem0ai` is not installed, the demo prints an install hint and runs with the
summary middleware only. The unit tests remain deterministic and network-free
without importing mem0.

## Experimental Path: Structured Memory

The `memory_agent/` package is a framework-light structured-memory experiment.
It keeps an append-only transcript, renders memory entries into the system
prompt after session-local selection, and uses a `MemoryUpdater` LLM to convert
evicted turns into operations:

- `ADD` creates a new memory entry.
- `UPDATE` refines or extends an entry that remains true.
- `SUPERSEDE` marks an entry inactive when it is contradicted or no longer true.
- `NOOP` records that nothing should be saved.

`demo_react.py` wires this memory layer into a LangChain ReAct agent through
`StructuredMemoryMiddleware`.

Run it with:

```bash
export OPENAI_API_KEY="your-api-key"
python demo_react.py
```

Use this path only after the simple summary version is not enough. It is useful
when you need provenance, conflict history, or stronger guarantees that failed
memory updates do not drop messages.

## Update Policy

For structured memory, `UPDATE` should not rewrite history. Use it only for
same-direction clarification, refinement, or extension.

When a user explicitly changes, reverses, or rejects an earlier preference,
decision, fact, goal, or plan, the updater prompt requires:

1. `SUPERSEDE` the old active entry.
2. `ADD` a new replacement entry.

This keeps the final state correct while preserving the audit trail.

## Tests

```bash
python -m pytest -q
```

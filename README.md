# MemoryAgent

This repo has three runnable paths:

1. `react_summary_agent.py` is the recommended first version. It uses
   LangChain's built-in `SummarizationMiddleware` with a simple ReAct-style
   tool-calling agent.
2. `react_hybrid_agent.py` uses `StructuredMemoryMiddleware` for auditable
   in-session compression and adds a mem0-backed long-term vector memory layer
   for cross-session recall.
3. `demo_react.py` is an experimental structured-memory version. It replaces
   `SummarizationMiddleware` with `StructuredMemoryMiddleware`, which evicts
   old messages into operation-based memory entries.

Start with the first path unless you specifically need cross-session semantic
recall, auditable memory entries, superseded history, or custom eviction
behavior.

For the full design, data flow, and memory policy details, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Project Layout

Runnable files at the repo root are thin entry points. Shared application
assembly lives inside the package:

- `memory_agent/config.py`: environment-backed config dataclasses.
- `memory_agent/demo_tools.py`: shared demo tools (`weather`, `calculator`).
- `memory_agent/agent_builders.py`: LangChain agent builders for summary,
  structured, and structured+mem0 paths.
- `memory_agent/memory.py`, `transcript.py`, `window.py`, `selector.py`,
  `updater.py`: framework-light structured-memory core.
- `memory_agent/langchain_middleware.py`: LangChain adapter for structured
  memory.
- `memory_agent/longterm.py`, `longterm_middleware.py`: mem0-backed long-term
  recall protocol and LangChain adapter.
- `scripts/run_beam_case.py`: BEAM one-case benchmark runner. It uses a
  `BeamRunConfig` object internally and supports `structured_mem0` and
  `raw_mem0` modes.
- `scripts/run_beam_case_deepagent.py`: deepagents variant of the BEAM runner.
  The answering stage uses `create_deep_agent` with a `search_long_term_memory`
  tool, so the agent performs its own (possibly multi-step) mem0 retrieval
  instead of receiving pre-retrieved top-k context. `deepagents` requires
  Python >= 3.11, so it lives in a separate venv:

  ```bash
  python3.12 -m venv .venv-deepagents
  .venv-deepagents/bin/pip install -r requirements-deepagents.txt
  .venv-deepagents/bin/python scripts/run_beam_case_deepagent.py
  ```

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
export THREAD_ID="react-summary-demo"
```

The demo uses `trigger=("messages", 6)` and `keep=("messages", 2)` so
summarization happens quickly in a local test. For real use, prefer a token
trigger such as `trigger=("tokens", 4000)`.

## Hybrid Path: Structured Memory + Long-Term Vector Memory (mem0)

`react_hybrid_agent.py` adds a third runtime path for conversations that need
both auditable short-term context compression and long-term recall.
`StructuredMemoryMiddleware` evicts older turns into operation-based memory
entries and injects them under `# Conversation Memory`. A second middleware,
placed after it, detects the message IDs that disappeared from the active
context, persists those raw turns to mem0, and injects semantically relevant
long-term memories under `# Long-Term Memory` on every model call.

Run it with:

```bash
export OPENAI_API_KEY="your-api-key"
python react_hybrid_agent.py
```

Optional knobs:

```bash
export MAIN_MODEL="openai:gpt-5.5"
export MEMORY_MODEL="openai:gpt-5.4-mini"
export STRUCTURED_MAX_TOKENS="220"
export STRUCTURED_MAX_MEMORY_TOKENS="600"
export STRUCTURED_KEEP_MESSAGES="4"
export MEM0_LLM_MODEL="gpt-4o-mini"
export MEM0_USER_ID="demo-user"
export MEM0_DATA_DIR=".mem0"
export THREAD_ID="react-hybrid-memory-demo"
```

The local mem0 store persists under `.mem0/`, including embedded Qdrant data
and mem0's history database. `MEM0_LLM_MODEL` controls mem0's fact-extraction
model separately from the main agent model. If `# Conversation Memory` and
`# Long-Term Memory` conflict, the demo system prompt tells the model to prefer
the structured conversation memory as the current state. Re-running the script
with the same `MEM0_USER_ID` and `MEM0_DATA_DIR` demonstrates cross-session
recall. Embedded Qdrant takes an exclusive file lock on its local data path, so
run only one process against the same `.mem0/` directory at a time.

If `mem0ai` is not installed, the demo prints an install hint and runs with the
structured middleware only. The unit tests remain deterministic and network-free
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

Preferences and goal entries are pinned, so they are always kept in injected
memory regardless of the token budget. The `exact_values` section preserves
numbers, dates, versions, identifiers, paths, and URLs verbatim. Tool outputs
are deterministically truncated before the updater LLM sees them
(`max_tool_turn_chars`, default 2000 chars), avoiding wasted tokens on
re-derivable output.

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

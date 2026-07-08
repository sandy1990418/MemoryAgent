# MemoryAgent

MemoryAgent is a small memory architecture playground for LangChain agents.
The package is organized by responsibility, not by "whatever file was created
first":

1. `summary/`: baseline LangChain `SummarizationMiddleware` path.
2. `structured/`: this repo's operation-based structured memory system.
3. `longterm/`: long-term recall integration for mem0-style vector memory.
4. `agents/`: runnable LangChain agent assembly for each path.

For the detailed component graph and data flow, see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Project Layout

```text
memory_agent/
  __init__.py              public package exports

  summary/                 summary-related code only
    agent.py               build_summary_agent using SummarizationMiddleware

  structured/              operation-based memory domain/runtime
    memory.py              Memory store and ADD/UPDATE/SUPERSEDE/NOOP ops
    transcript.py          append-only transcript
    window.py              framework-free sliding context window
    selector.py            prompt memory selection
    updater.py             LLM-driven memory operation generator
    session.py             framework-free structured chat session
    middleware.py          LangChain StructuredMemoryMiddleware

  longterm/                long-term recall integration
    middleware.py          LangChain LongTermMemoryMiddleware

  agents/                  application assembly
    common.py              thread/invoke/printing helpers
    structured.py          structured-memory agent builder
    hybrid.py              structured + long-term mem0 agent builder

  clients/                 external service boundaries
    llm.py                 LLMClient protocol, OpenAIClient adapter
    mem0.py                LongTermMemory protocol, Mem0 adapter

  models/                  dataclasses, configs, constants
    config.py              .env-backed config models
    sections.py            SectionConfig and default section lists
    memory.py              MemoryEntry, SelectedMemory
    transcript.py          Turn
    longterm.py            LongTermHit
    runtime.py             agent runtime containers
    beam.py                BEAM runner models

  tools/
    demo.py                demo weather/calculator tools
```

Root runnable files stay thin:

```text
react_summary_agent.py     uses memory_agent.summary
demo_react.py              uses memory_agent.agents.structured
react_hybrid_agent.py      uses memory_agent.agents.hybrid
demo.py                    uses memory_agent.structured directly
```

## Setup

Main demos use the default project venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set at least:

```bash
OPENAI_API_KEY="your-api-key"
```

Run the baseline summary path:

```bash
python react_summary_agent.py
```

Run structured memory:

```bash
python demo_react.py
```

Run structured memory plus mem0:

```bash
python react_hybrid_agent.py
```

The DeepAgents BEAM runner is optional and requires Python >= 3.11. Do not
install `requirements-deepagents.txt` into the main Python 3.10 `.venv`; pip
will reject `deepagents` with a `Requires-Python >=3.11` error. Use a separate
venv:

```bash
python3.12 -m venv .venv-deepagents
.venv-deepagents/bin/pip install -r requirements-deepagents.txt
.venv-deepagents/bin/python scripts/run_beam_case_deepagent.py
```

BEAM runners report a cheap heuristic rubric score, run a BEAM-style LLM judge
by default, and always write a BEAM-compatible answers JSON next to the
detailed trace. The judge model defaults to `BEAM_JUDGE_MODEL`, then
`BEAM_ANSWER_MODEL`, then `gpt-5.4-nano`. Override it with `--judge-model`, or
disable judge calls with `--no-judge`:

```bash
python scripts/run_beam_case.py

python scripts/run_beam_case.py --no-judge

.venv-deepagents/bin/python scripts/run_beam_case_deepagent.py \
  --judge-model gpt-5.4-nano
```

The detailed output JSON then includes `heuristic_rubric_rate`,
`judge_rubric_rate`, and BEAM-style `judge_score`, plus per-question
`judge_checks` with `score` and `reason`. With judge enabled, the runner also
writes an `evaluation-*.json` file shaped like BEAM's evaluator output, with
`llm_judge_score` and `llm_judge_responses`.

To smoke-test a few downloaded BEAM cases from `BEAM/chats/100K` using only
structured summary memory (no mem0 ingestion/retrieval), run:

```bash
python scripts/run_beam_cases.py --max-cases 3
```

Use `--case-ids 1 2 3` for explicit cases. Results are written under
`data/beam/results/100K/<case_id>/`, plus a `batch_manifest_*.json` summary.
By default, the runner evaluates `contradiction_resolution`,
`knowledge_update`, `preference_following`, `instruction_following`,
`abstention`, and `summarization`, with all questions in those abilities.
Pass `--all-question-types` for the complete BEAM suite, or
`--max-questions-per-type 1` for a faster smoke test.

## Configuration

All runnable demos call `load_project_env()`, which loads `.env` from the repo
root. `.env.example` documents the supported variables.

Common variables:

```bash
MAIN_MODEL="openai:gpt-5.4-nano"
SUMMARY_MODEL="openai:gpt-5.4-nano"
MEMORY_MODEL="openai:gpt-5.4-nano"
THREAD_ID="react-summary-demo"
STRUCTURED_MAX_TOKENS="600"
STRUCTURED_MAX_MEMORY_TOKENS="600"
STRUCTURED_KEEP_MESSAGES="4"
MEMORY_PROFILE="practical"
```

`MEMORY_PROFILE` separates sparse product retention from detailed evaluation:
`practical` defaults ordinary Q&A to NOOP, `agent` retains richer execution
state, and `eval` (or `beam`) enables detail-heavy BEAM extraction.

`LLMClient` and `OpenAIClient` are intentionally different:

- `LLMClient` is a small protocol used by core code and tests. Anything with
  `complete(system, messages, model=None) -> str` can satisfy it.
- `OpenAIClient` is the real adapter backed by `langchain_openai.ChatOpenAI`.
  It is one implementation of `LLMClient`, not a duplicate abstraction.

## mem0 Modes

`react_hybrid_agent.py` uses `HybridAgentConfig` and supports three backends:

```bash
# Local development/test mode. Uses embedded Qdrant under .mem0/.
MEM0_BACKEND="local"
MEM0_DATA_DIR=".mem0"
MEM0_USER_ID="demo-user"
MEM0_LLM_MODEL="gpt-5.4-nano"

# Hosted/custom mem0 content. MEM0_DATA_DIR is ignored.
MEM0_BACKEND="platform"
MEM0_API_KEY="your-mem0-key"
MEM0_USER_ID="your-user-id"

# Structured memory only.
MEM0_BACKEND="disabled"
```

For your own mem0 data, set `MEM0_BACKEND=platform`, `MEM0_API_KEY`, and
`MEM0_USER_ID`; you do not need `MEM0_DATA_DIR`. For local testing, leave
`MEM0_BACKEND=local` and use the default `.mem0` directory, or point
`MEM0_DATA_DIR` at a temporary test store.

Embedded Qdrant takes an exclusive file lock on the local data path, so run only
one process against the same local `MEM0_DATA_DIR` at a time.

## Structured Memory Policy

`MemoryUpdater` turns evicted turns into operations:

- `ADD`: create a new memory entry.
- `UPDATE`: refine an active entry that remains true.
- `SUPERSEDE`: mark an active entry inactive when contradicted.
- `NOOP`: preserve nothing from that batch.

Updater-generated batches are applied atomically. If parsing, provenance
validation, or operation application fails, source messages stay in context for
retry instead of being dropped.

## Tests

```bash
python -m pytest -q
```

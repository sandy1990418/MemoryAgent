# MemoryAgent

MemoryAgent is a general-purpose, token-efficient memory system for both
conversational and agentic workloads. The current development phase focuses on
chat memory, using selected BEAM capabilities to evaluate memory correctness,
update behavior, retrieval quality, and performance under constrained token
budgets. The common domain remains extensible to task state, execution history,
tool observations, decisions, failures, artifacts, and reusable experience.

BEAM results validate the current **chat profile only**; they do not establish
complete agent-memory capability.

1. `core/`, `update/`, and `retrieval/`: operation-based structured memory.
2. `longterm/`: long-term recall integration for mem0-style vector memory.
3. `agents/`: reusable LangChain agent assembly.
4. `demos/`: runnable examples, baseline summary code, and demo tools.

For the detailed component graph and data flow, see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Project Layout

```text
memory_agent/
  __init__.py              public package exports

  core/                    entries, schema, store, transcript, working window
  policies/                structured and event-memory policy contracts
  normalization/           injectable workload-aware normalization
  update/                  extraction, prompts, validation, compaction
  retrieval/               answer selection, rendering, quality signals
  application/             chat/session and structured/event-memory services
  adapters/events/         chat and agent-trace event adapters
  adapters/langchain/      StructuredMemoryMiddleware framework adapter

  domain/                  generic event-memory data contracts

  longterm/                long-term recall integration
    middleware.py          LangChain LongTermMemoryMiddleware

  agents/                  application assembly
    common.py              thread/invoke/printing helpers
    structured.py          structured-memory agent builder
    hybrid.py              structured + long-term mem0 agent builder

  clients/                 external service boundaries
    llm.py                 LLMClient protocol, OpenAIClient adapter
    mem0.py                LongTermMemory protocol, Mem0 adapter

  models/                  remaining integration/config/runtime DTOs
    config.py              .env-backed config models
    longterm.py            LongTermHit
    runtime.py             agent runtime containers
```

Evaluation tooling stays outside the installable runtime package:

```text
evaluation/
  memory/                  replay, metrics, manifests, report schemas
  beam/                    BEAM-specific adapters, routing, snapshots, reports
```

Package code imports directly from `core`, `policies`, `update`, `retrieval`,
`application`, or `adapters`; the previous `structured` and `profiles`
packages have been removed.

Demo-only code stays outside the importable product package:

```text
demos/
  summary.py               SummarizationMiddleware baseline builder
  config.py                demo-only environment configs
  tools.py                 weather/calculator demo tools
  summary_agent.py         summary baseline entry point
  structured_agent.py      structured-memory agent entry point
  hybrid_agent.py          structured + mem0 entry point
  manual_session.py        framework-free legacy session example
```

`memory_agent/agents` accepts tools through dependency injection and has no
dependency on `demos/`.

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
python -m demos.summary_agent
```

Run structured memory:

```bash
python -m demos.structured_agent
```

Run structured memory plus mem0:

```bash
python -m demos.hybrid_agent
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
`llm_judge_score` and `llm_judge_responses`. The detailed trace also records
the source commit, resolved run config, and token usage for `updater`,
`compactor`, `agent`, and `judge` roles.

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

Product memory defaults live in `configs/product.yaml`; BEAM dataset,
abilities, judge, and question-cap defaults live in `configs/beam.yaml`.
Environment variables override individual YAML values, and CLI arguments
override the resolved BEAM defaults. Use `--beam-config` to select another
BEAM YAML file. Runnable demos also load `.env` from the repo root.

Common variables:

```bash
MAIN_MODEL="openai:gpt-5.4-nano"
SUMMARY_MODEL="openai:gpt-5.4-nano"
MEMORY_MODEL="openai:gpt-5.4-nano"
THREAD_ID="react-summary-demo"
STRUCTURED_MAX_TOKENS="600"
STRUCTURED_MAX_MEMORY_TOKENS="600"
STRUCTURED_KEEP_MESSAGES="4"
MEMORY_PROFILE="chat"
MEMORY_SECTIONS="chat"
MEMORY_COMPACTION_THRESHOLD="30"
```

`MEMORY_PROFILE` separates workloads: `chat` is the product default and keeps
durable conversational context while dropping ordinary Q&A; `practical` is a
compatibility alias for the earlier chat behavior; `agent` is an extension
profile for execution state; and `eval` (or runner-only `beam`) is a broad
legacy evaluation profile.

There are two deliberately named policy contracts during the agent-event
transition:

- `StructuredMemoryPolicy` configures the production operation-based runtime.
- `EventMemoryPolicy` classifies generic `MemoryEvent` objects at the future
  agent-event ingestion boundary.

Only the explicit structured/event names are exposed; the ambiguous legacy
policy and service aliases have been removed.

Fixed-budget comparisons are implemented in
`evaluation.beam.compare_fixed_budget_runs`. They require identical cases,
questions, variants, and context budgets, separate production tokens from judge
tokens, flag context-budget violations, and label results as chat-only evidence.

`LLMClient` and `OpenAIClient` are intentionally different:

- `LLMClient` is a small protocol used by core code and tests. Anything with
  `complete(system, messages, model=None) -> str` can satisfy it.
- `OpenAIClient` is the real adapter backed by `langchain_openai.ChatOpenAI`.
  It is one implementation of `LLMClient`, not a duplicate abstraction.

## mem0 Modes

`demos.hybrid_agent` uses `HybridAgentConfig` and supports three backends:

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

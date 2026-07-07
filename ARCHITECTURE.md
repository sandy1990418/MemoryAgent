# MemoryAgent Architecture

This repo follows two practical rules:

- Put importable package code in clear packages with explicit responsibility.
  This follows the direction of the Python Packaging User Guide's package
  layout guidance: import packages should be obvious and separated from repo
  tooling/files.
- Treat LangChain middleware as composable context-management units. LangChain
  documents middleware as the mechanism for prompt/context transformation,
  retries, guardrails, and other agent-loop hooks, so summary, structured
  memory, and long-term recall should be separate middleware concerns.

References:

- Python Packaging User Guide: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- LangChain middleware overview: https://docs.langchain.com/oss/python/langchain/middleware
- LangChain agents/context management: https://docs.langchain.com/oss/python/langchain/agents

## Package Map

```text
memory_agent/
  summary/       SummaryMiddleware baseline only.
  structured/    Operation-based memory domain and runtime.
  longterm/      Long-term vector recall integration.
  agents/        Agent assembly for runnable demos.
  clients/       External service protocols/adapters.
  models/        Dataclasses, config models, constants.
  tools/         Demo tools.
```

The key distinction:

```text
summary/
  Uses LangChain SummarizationMiddleware.
  It is a baseline compression strategy.
  It does not produce structured memory entries.

structured/
  Owns Memory, Transcript, WorkingWindow, MemorySelector, MemoryUpdater,
  MemorySession, and StructuredMemoryMiddleware.
  It is not summary code. It stores auditable entries through operations.

longterm/
  Owns LongTermMemoryMiddleware.
  It integrates long-term recall with the agent loop.
  The concrete mem0 adapter lives in clients/mem0.py.
```

## High-Level Component Graph

```mermaid
flowchart TD
    Entry[Runnable scripts] --> SummaryScript[react_summary_agent.py]
    Entry --> StructuredScript[demo_react.py]
    Entry --> HybridScript[react_hybrid_agent.py]
    Entry --> PlainScript[demo.py]

    SummaryScript --> SummaryPkg[memory_agent.summary]
    StructuredScript --> StructuredAgent[memory_agent.agents.structured]
    HybridScript --> HybridAgent[memory_agent.agents.hybrid]
    PlainScript --> StructuredPkg[memory_agent.structured]

    SummaryPkg --> LC1[LangChain create_agent]
    SummaryPkg --> LCSum[SummarizationMiddleware]

    StructuredAgent --> StructuredPkg
    StructuredPkg --> Store[Memory]
    StructuredPkg --> Transcript[Transcript]
    StructuredPkg --> Window[WorkingWindow]
    StructuredPkg --> Selector[MemorySelector]
    StructuredPkg --> Updater[MemoryUpdater]
    StructuredPkg --> StructuredMW[StructuredMemoryMiddleware]

    HybridAgent --> StructuredAgent
    HybridAgent --> LongTermPkg[memory_agent.longterm]
    LongTermPkg --> LongTermMW[LongTermMemoryMiddleware]

    Updater --> LLMProtocol[LLMClient protocol]
    LLMProtocol --> OpenAI[OpenAIClient adapter]

    LongTermMW --> LongTermProtocol[LongTermMemory protocol]
    LongTermProtocol --> Mem0[Mem0LongTermMemory adapter]

    Store --> Models[memory_agent.models]
    Transcript --> Models
    Selector --> Models
```

## Dependency Direction

```mermaid
flowchart LR
    Models[models] --> Structured[structured]
    Models --> LongTerm[longterm]
    Models --> Agents[agents]
    Clients[clients] --> Structured
    Clients --> LongTerm
    Structured --> Agents
    LongTerm --> Agents
    Tools[tools] --> Agents
    Summary[summary] --> Agents
```

Rules:

- `models/` has no LangChain, mem0, or OpenAI imports.
- `clients/` is where external services are adapted behind small protocols.
- `structured/` owns structured-memory behavior and may depend on
  `LLMClient`, but not on a concrete OpenAI import except through injection.
- `longterm/` owns LangChain long-term recall middleware; concrete mem0 details
  stay in `clients/mem0.py`.
- `summary/` owns only the LangChain built-in summary baseline.
- `agents/` wires these pieces together for runnable apps.

## Runtime Path 1: Summary Baseline

Entry point: `react_summary_agent.py`

Code location:

```text
memory_agent/summary/agent.py
```

```mermaid
sequenceDiagram
    participant U as User
    participant A as LangChain Agent
    participant S as SummarizationMiddleware
    participant C as InMemorySaver
    participant M as Model

    U->>A: user message
    A->>S: middleware hook
    S->>C: compact old messages when triggered
    A->>M: model/tool loop
    M-->>A: assistant response
    A-->>U: response
```

This is the only summary-specific package. If you are looking for summary
behavior, start at `memory_agent.summary`.

## Runtime Path 2: Structured Memory With LangChain

Entry point: `demo_react.py`

Code locations:

```text
memory_agent/agents/structured.py
memory_agent/structured/middleware.py
memory_agent/structured/updater.py
memory_agent/structured/memory.py
```

```mermaid
sequenceDiagram
    participant U as User
    participant A as LangChain Agent
    participant SM as StructuredMemoryMiddleware
    participant T as Transcript
    participant Up as MemoryUpdater
    participant Mem as Memory
    participant Sel as MemorySelector
    participant L as LLMClient

    U->>A: user message
    A->>SM: before_model(state)
    SM->>T: mirror unseen messages to Turns
    SM->>SM: check token budget
    alt over budget
        SM->>SM: choose safe cutoff without splitting tool pairs
        SM->>Up: evicted turns + current memory
        Up->>L: request ADD/UPDATE/SUPERSEDE/NOOP JSON
        L-->>Up: ops JSON
        Up->>Mem: validate and apply atomically
        alt success
            SM-->>A: remove evicted messages
        else failure or rejected ops
            SM-->>A: keep all messages for retry
        end
    else within budget
        SM-->>A: no state update
    end
    A->>SM: wrap_model_call(request)
    SM->>Sel: select active memory for latest query
    Sel->>Mem: read active entries
    SM->>A: request with # Conversation Memory
```

This path is not summary-based. It converts evicted turns into addressable
memory entries and preserves superseded state.

## Runtime Path 3: Structured Memory + Long-Term mem0

Entry point: `react_hybrid_agent.py`

Code locations:

```text
memory_agent/agents/hybrid.py
memory_agent/longterm/middleware.py
memory_agent/clients/mem0.py
```

```mermaid
flowchart TD
    A[User message] --> B[StructuredMemoryMiddleware]
    B --> C[# Conversation Memory]
    B --> D[Old messages evicted after successful update]
    D --> E[LongTermMemoryMiddleware tracks disappeared message IDs]
    E --> F[LongTermMemory.add]
    F --> G{MEM0_BACKEND}
    G -->|local| H[Mem0 OSS local store at MEM0_DATA_DIR]
    G -->|platform| I[Hosted/custom mem0 through MEM0_API_KEY]
    G -->|disabled| J[Skip long-term recall]
    E --> K[LongTermMemory.search latest user query]
    K --> L[# Long-Term Memory]
    C --> M[Model call]
    L --> M
```

`# Conversation Memory` is current structured state. `# Long-Term Memory` is
supporting recall from older stored content.

## Runtime Path 4: Framework-Free Structured Session

Entry point: `demo.py`

Code locations:

```text
memory_agent/structured/session.py
memory_agent/structured/window.py
memory_agent/structured/transcript.py
```

```mermaid
flowchart TD
    A[User text] --> B[Transcript.append user Turn]
    B --> C[WorkingWindow.add]
    C --> D{Prompt would exceed budget?}
    D -->|yes| E[WorkingWindow.eviction_batch]
    E --> F[MemoryUpdater.update]
    F --> G{Ops accepted?}
    G -->|yes| H[Memory.apply_ops_atomically]
    H --> I[WorkingWindow.remove evicted Turns]
    G -->|no| J[Keep Turns for retry]
    D -->|no| K[Build prompt]
    I --> K
    J --> K
    K --> L[MemorySelector.select]
    L --> M[Memory.render]
    M --> N[LLMClient.complete]
    N --> O[Transcript.append assistant Turn]
    O --> P[WorkingWindow.add assistant Turn]
```

## Data Model Relationships

```mermaid
classDiagram
    class SectionConfig {
      key: str
      prefix: str
      title: str
      description: str
    }

    class Turn {
      id: int
      role: str
      content: str
    }

    class MemoryEntry {
      id: str
      section: str
      text: str
      provenance: list[int]
      status: active|superseded
      note: str
    }

    class SelectedMemory {
      entry: MemoryEntry
      score: float
      reasons: tuple[str]
    }

    class LongTermHit {
      text: str
      score: float?
      metadata: dict?
    }

    class Memory {
      entries: dict[str, MemoryEntry]
      narrative: str
      apply_ops()
      apply_ops_atomically()
      render()
    }

    class Transcript {
      append()
      get()
      all()
    }

    class MemorySelector {
      select()
      select_with_scores()
    }

    SectionConfig --> Memory
    Turn --> Transcript
    MemoryEntry --> Memory
    MemoryEntry --> SelectedMemory
    Memory --> MemorySelector
    LongTermHit --> LongTermMemoryMiddleware
```

## Memory Update Contract

```mermaid
flowchart TD
    A[Evicted Turns] --> B[Build updater prompt]
    C[Current Memory including superseded entries] --> B
    D[SectionConfig list] --> B
    B --> E[LLMClient.complete]
    E --> F{Parse JSON array?}
    F -->|no| G[UpdateFailed]
    F -->|yes| H[Normalize common mistakes]
    H --> I{Provenance valid?}
    I -->|no| J[Reject batch]
    I -->|yes| K{Apply on copy?}
    K -->|any op invalid| J
    K -->|all valid| L[Commit candidate Memory]
    G --> M[Caller keeps source messages]
    J --> M
    L --> N[Caller may evict source messages]
```

Supported operations:

- `ADD`: create a new active entry.
- `UPDATE`: refine an active entry that remains true.
- `SUPERSEDE`: mark an active entry inactive when contradicted.
- `NOOP`: save nothing from the batch.

The Python layer validates shape, IDs, sections, and provenance. The LLM still
owns the semantic choice.

Memory quality policy:

- Entries should be atomic and concise.
- Exact dates, versions, counts, durations, percentages, latencies, endpoints,
  table/column names, file names, error messages, library names, and deployment
  targets should be preserved in `exact_values` when they may matter later.
- Generic assistant advice and example code should not become memory unless the
  user accepts, decides, implements, observes, or reports it.
- `open_questions` is only for unresolved blockers or decisions that remain
  important after the turn. Ordinary one-off help requests should usually be
  `NOOP` or become concise `facts`/`progress` only when they contain durable
  state.
- `status_changes` captures contradictions, corrections, reversals, denials,
  and latest-vs-previous truths. This section is intentionally rendered before
  generic facts so contradiction-resolution questions can see it.
- `timeline` captures ordered milestones and event sequences. This section is
  intentionally rendered before generic facts/open questions so chronology
  questions can see it.

BEAM answering uses question-specific structured-memory selection before
building the answer context. This prevents generic facts and open questions
from crowding out status-change or timeline entries that are relevant to a
specific probing question.

## Configuration Flow

```mermaid
flowchart LR
    A[.env] --> B[load_project_env]
    B --> C[models/config.py]
    C --> D[summary/agent.py]
    C --> E[agents/structured.py]
    C --> F[agents/hybrid.py]
    F --> G{MEM0_BACKEND}
    G -->|local| H[Mem0LongTermMemory.from_local]
    G -->|platform| I[Mem0LongTermMemory.from_platform]
    G -->|disabled| J[No LongTermMemoryMiddleware]
```

Important mem0 settings:

```text
MEM0_BACKEND=local
  Uses MEM0_DATA_DIR, default .mem0.
  Good for local testing and repeatable demos.

MEM0_BACKEND=platform
  Uses MEM0_API_KEY and MEM0_USER_ID.
  Does not require or use MEM0_DATA_DIR.
  Use this for your own hosted/custom mem0 content.

MEM0_BACKEND=disabled
  Skips long-term vector memory.
  Structured in-session memory still runs.
```

`build_hybrid_agent(..., long_term_memory=...)` accepts an injected
`LongTermMemory` implementation for tests or custom adapters.

## LLM Boundary

```mermaid
flowchart TD
    A[MemoryUpdater / MemorySession] --> B[LLMClient protocol]
    B --> C[OpenAIClient]
    B --> D[Test fake]
    B --> E[Future provider adapter]
```

`LLMClient` exists so core code can depend on a tiny behavior contract. It is
why tests can use deterministic fakes without importing LangChain OpenAI.
`OpenAIClient` is the production adapter that satisfies that protocol.

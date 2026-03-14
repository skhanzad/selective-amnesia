# Selective Amnesia

Graph-structured external memory for LLM agents with selective forgetting.

## Research Question

> Does a graph-structured, human-editable external memory with selective forgetting outperform flat memory and vector retrieval baselines on long-term memory tasks, especially under contradiction, drift, and memory budget pressure?

This project builds a **graph-based external memory layer** for LLM agents — not a visualization of the LLM's internal state — and tests whether intelligent forgetting policies improve performance, controllability, and debuggability compared to naive baselines.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LangGraph Pipeline                     │
│                                                             │
│   User Query                                                │
│       │                                                     │
│       ▼                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐  │
│   │ Retrieve │───▶│ Generate │───▶│ Extract  │───▶│Forget│  │
│   │ Memories │    │ Response │    │ Memories │    │      │  │
│   └──────────┘    └──────────┘    └──────────┘    └──────┘  │
│       ▲                                │              │     │
│       │                                ▼              ▼     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Memory Graph (NetworkX)                │   │
│   │  Typed nodes · Typed edges · Soft-delete · JSON I/O │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

Each conversation turn flows through four stages:

1. **Retrieve** — query the memory graph for relevant context (keyword similarity + neighbor expansion)
2. **Generate** — produce a response using the LLM with retrieved memories injected into the prompt
3. **Extract** — use the LLM to identify new facts, preferences, events, and entities from the turn
4. **Forget** — apply a forgetting policy to prune the graph when it exceeds a memory budget

## Project Structure

```
selective-amnesia/
├── src/
│   ├── main.py                          # Interactive CLI entrypoint
│   ├── memory/
│   │   ├── schemas.py                   # Pydantic models (MemoryNode, MemoryEdge, etc.)
│   │   ├── graph_store.py               # NetworkX-based CRUD for the memory graph
│   │   ├── forgetting.py                # Modular forgetting policies
│   │   └── retriever.py                 # Memory retrieval strategies
│   ├── agent/
│   │   ├── state.py                     # LangGraph state schema
│   │   ├── graph.py                     # Build & compile the LangGraph pipeline
│   │   └── nodes.py                     # Pipeline node functions
│   ├── evaluation/
│   │   ├── metrics.py                   # F1, exact match, memory stats
│   │   ├── data_loaders.py              # LoCoMo & LongMemEval dataset loaders
│   │   └── runner.py                    # Benchmark runner & experiment orchestration
│   └── ui/
│       └── app.py                       # Streamlit graph explorer & editor
├── configs/
│   ├── base.yaml                        # Default configuration
│   └── experiments/
│       ├── b0_no_memory.yaml            # Baseline: no external memory
│       ├── b1_flat_memory.yaml          # Baseline: flat recency-sorted list
│       ├── b3_graph_no_forgetting.yaml  # Graph memory, no forgetting
│       └── b4_graph_with_forgetting.yaml# Graph memory + hybrid forgetting
├── scripts/
│   └── run_benchmark.py                 # Benchmark suite orchestration
├── tests/
│   ├── test_memory.py                   # Unit tests for memory operations
│   └── test_evaluation.py              # Unit tests for metrics & loaders
├── ext/                                 # Git submodules for benchmark datasets
│   ├── LongMemEval/
│   └── locomo/
├── data/                                # Saved memory graphs (JSON)
├── results/                             # Experiment result files
├── pyproject.toml
├── .env.example
└── AGENTS.md                            # Original project brief
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.com/) running locally (default LLM backend), or an OpenAI API key

### Installation

```bash
# Clone with submodules (for benchmark datasets)
git clone --recurse-submodules https://github.com/<your-user>/selective-amnesia.git
cd selective-amnesia

# Install dependencies
uv sync

# If using OpenAI instead of Ollama
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

If using Ollama, make sure it's running and the model is pulled:

```bash
ollama pull llama3.2
```

## Usage

### Interactive CLI

Start a conversation with the memory-augmented agent:

```bash
uv run python src/main.py
```

Options:

```bash
uv run python src/main.py --config configs/experiments/b4_graph_with_forgetting.yaml
uv run python src/main.py --load data/memory_graph.json
uv run python src/main.py --debug
```

In-session commands:

| Command      | Description                      |
|-------------|-----------------------------------|
| `/memories` | List all stored memory nodes      |
| `/stats`    | Show memory type distribution     |
| `/save`     | Save current graph to disk        |
| `/clear`    | Clear the memory graph            |

Each turn displays a status line:

```
You: What's my favorite coffee?
Assistant: You mentioned you prefer dark roast espresso.
  [mem: 45 | retrieved: 3 | +2 | -1]
```

Where `mem` = total nodes, `retrieved` = memories used this turn, `+N` = new memories extracted, `-N` = memories forgotten.

### Graph Explorer UI

Launch the Streamlit-based memory graph visualizer and editor:

```bash
uv run streamlit run src/ui/app.py
```

Features:
- **Interactive graph** — PyVis visualization with nodes colored by type and sized by importance
- **Node table** — sortable list of all memories with metadata
- **Edit/Delete** — modify node content, type, importance; toggle enabled/disabled; delete permanently
- **Add nodes & edges** — create new memories and relationships
- **Filter** — by node type or edge type
- **Save/Load** — persist changes to disk

### Running Benchmarks

Run the experimental suite against the LoCoMo and LongMemEval datasets:

```bash
# Quick smoke test (1 sample, 10 QA pairs)
uv run python scripts/run_benchmark.py --dataset locomo --max-samples 1 --max-qa 10

# Full suite with all baselines on both datasets
uv run python scripts/run_benchmark.py --dataset all

# Specific baselines only
uv run python scripts/run_benchmark.py --dataset longmemeval --baselines b0 b4

# Quiet mode (less output)
uv run python scripts/run_benchmark.py --dataset locomo --quiet

# Custom results directory
uv run python scripts/run_benchmark.py --results-dir my_results/
```

Results are saved as JSON to `results/` with timestamps.

### Running Tests

```bash
uv run pytest tests/ -v
```

## Memory System

### Node Types

Memories are stored as typed nodes in a NetworkX graph:

| Type              | Description                                |
|-------------------|--------------------------------------------|
| `entity`          | A person, place, organization, or thing    |
| `fact`            | A factual statement or piece of knowledge  |
| `event`           | Something that happened at a point in time |
| `user_preference` | A user's stated preference or opinion      |
| `task`            | A task, goal, or action item               |
| `summary`         | A compressed summary of other memories     |
| `source`          | An information source or reference         |
| `belief`          | A belief or assumption (may be uncertain)  |

### Edge Types

Relationships between memories are typed edges:

| Type              | Description                                    |
|-------------------|------------------------------------------------|
| `related_to`      | General relationship                           |
| `refers_to`       | One memory references another                  |
| `supports`        | Evidence or corroboration                      |
| `contradicts`     | Direct contradiction between memories          |
| `supersedes`      | Newer memory replaces an older one             |
| `caused_by`       | Causal relationship                            |
| `temporal_before` | Temporal ordering                              |
| `derived_from`    | One memory was derived from another            |
| `similar_to`      | Semantic similarity                            |

### Forgetting Policies

Forgetting is implemented as modular, swappable policies. Each policy scores nodes from 0.0 (keep) to 1.0 (forget):

| Policy       | Strategy                                                                |
|-------------|--------------------------------------------------------------------------|
| `none`      | Never forget anything                                                    |
| `recency`   | Forget old, unaccessed nodes using exponential decay                     |
| `importance`| Forget low-importance nodes first                                        |
| `hybrid`    | Weighted combination: recency (40%) + importance (40%) + access frequency (20%) |

All policies respect a configurable `min_importance_to_keep` threshold — nodes above it are never forgotten regardless of score.

### Retrieval Strategies

| Strategy | Description                                                              |
|----------|--------------------------------------------------------------------------|
| `none`   | No external memory (pure LLM baseline)                                   |
| `flat`   | Return all memories sorted by recency                                    |
| `graph`  | Keyword relevance scoring (Jaccard similarity) + neighbor expansion + connectivity boosting |

## Configuration

All behavior is controlled via YAML config files. The base config (`configs/base.yaml`):

```yaml
llm:
  provider: "ollama"        # "ollama" or "openai"
  model: "llama3.2"         # e.g. "gpt-4o-mini" for openai
  temperature: 0.7

memory:
  max_nodes: 200
  default_importance: 0.5
  decay_rate: 0.05

retrieval:
  mode: "graph"             # "none", "flat", "graph"
  max_results: 10
  neighbor_depth: 1
  include_edge_context: true

forgetting:
  policy: "hybrid"          # "none", "recency", "importance", "hybrid"
  budget_target: 150
  recency_weight: 0.4
  importance_weight: 0.4
  access_frequency_weight: 0.2
  min_importance_to_keep: 0.3
  run_every_n_turns: 1
```

Experiment-specific configs in `configs/experiments/` override these defaults for each baseline.

## Experimental Design

### Baselines

| ID | Name                    | Retrieval | Forgetting | Budget |
|----|-------------------------|-----------|------------|--------|
| B0 | No external memory      | none      | none       | —      |
| B1 | Flat memory list        | flat      | none       | 500    |
| B3 | Graph, no forgetting    | graph     | none       | 500    |
| B4 | Graph + hybrid forgetting | graph   | hybrid     | 100    |

### Datasets

- **[LoCoMo](https://github.com/snap-stanford/locomo)** — Multi-session dialogue benchmark with 5 question categories: multi-hop, temporal, contextual, open-domain, adversarial
- **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)** — Long-context memory evaluation with oracle variant (evidence-only sessions)

Both are included as git submodules in `ext/`.

### Metrics

**Answer Quality:**

| Metric            | Description                                                         |
|-------------------|---------------------------------------------------------------------|
| Token F1          | Token-level F1 score after normalization                            |
| Exact Match       | Binary match after normalization                                    |
| Contains Match    | Whether the ground truth is contained in the prediction             |
| Multi-hop F1      | F1 specifically for multi-hop reasoning questions                   |
| Task Success Rate | Fraction of questions answered above F1 threshold (default 0.5)     |
| Per-category F1   | F1 broken down by question type                                     |

**Retrieval Quality:**

| Metric    | Description                                                                |
|-----------|----------------------------------------------------------------------------|
| Recall@k  | Fraction of top-k retrieved memories relevant to the ground truth answer   |
| MRR       | Mean Reciprocal Rank — 1/position of the first relevant retrieved memory   |

**Memory Operations:**

| Metric         | Description                                                            |
|----------------|------------------------------------------------------------------------|
| Edit Success   | After editing a memory, does the system's answer reflect the change?   |
| Delete Success | After deleting a memory, does the system stop referencing it?          |
| Locality Score | Do unrelated answers remain stable after an edit or deletion?          |

**Latency & Storage:**

| Metric             | Description                                                |
|--------------------|------------------------------------------------------------|
| Avg QA Latency     | Average time per question (retrieval + generation)         |
| Storage Bytes      | Total serialized size of the memory graph                  |
| Avg Node Size      | Average content size per memory node in bytes              |
| Memory Stats       | Node count, edge count, forgotten count, type distribution |
| Timing             | Ingest time, evaluation time, total experiment time        |

## Extending the Project

### Adding a New Forgetting Policy

1. Create a new class in `src/memory/forgetting.py` implementing the `ForgettingPolicy` protocol:

```python
class MyPolicy:
    def score(self, node: MemoryNode, current_turn: int) -> float:
        """Return 0.0 (keep) to 1.0 (forget)."""
        ...
```

2. Register it in `get_forgetting_policy()` in the same file.
3. Reference it by name in your config YAML under `forgetting.policy`.

### Adding a New Retriever

1. Create a new class in `src/memory/retriever.py` implementing the retriever interface:

```python
class MyRetriever:
    def retrieve(self, query: str, store: MemoryGraphStore,
                 current_turn: int, max_results: int) -> list[RetrievedMemory]:
        ...
```

2. Register it in `get_retriever()` in the same file.
3. Reference it by name in your config YAML under `retrieval.mode`.

### Changing Node or Edge Schemas

Edit the `NodeType` and `EdgeType` enums in `src/memory/schemas.py`. The Pydantic models will enforce the new types throughout the system.

## Assumptions and Limitations

### Assumptions

- An LLM can reliably extract structured memories from conversation turns via JSON prompting
- Token-level Jaccard similarity is a reasonable proxy for semantic relevance (until embeddings are added)
- Memory importance can be meaningfully estimated by the LLM at extraction time
- Forgetting policies operating on metadata (recency, importance, access count) can approximate human-like selective memory

### Current Limitations

- **No embedding-based retrieval** — uses token Jaccard similarity; a vector backend is stubbed but not implemented
- **LLM-dependent extraction** — memory quality depends on the LLM's ability to output valid JSON; malformed responses are handled gracefully but result in missed memories
- **Within-turn edges only** — edge creation is limited to relationships between memories extracted in the same turn; no cross-session edge inference
- **No active contradiction resolution** — contradictions are detected via edge types but not automatically resolved
- **No memory evolution replay** — the UI shows current state but doesn't support replaying how the graph changed over time
- **Single-user** — no multi-user or multi-agent memory separation

## Tech Stack

| Component       | Library                           |
|-----------------|-----------------------------------|
| Agent pipeline  | LangGraph                         |
| LLM interface   | LangChain (Ollama / OpenAI)       |
| Graph storage   | NetworkX                          |
| Data validation | Pydantic v2                       |
| Configuration   | PyYAML                            |
| Web UI          | Streamlit + PyVis                 |
| Testing         | pytest                            |
| Package manager | uv (hatchling build backend)      |

## License

This is a research prototype. See the repository for license details.

You are acting as a senior ML research engineer, research scientist, and product-minded prototyper.

I want you to HELP ME BUILD a local-first, modular research prototype for this idea:

PROJECT IDEA
Build an external memory system for an LLM agent that looks and behaves like an Obsidian-style graph: nodes, links, clusters, evolving memory, and editable relationships. The goal is NOT to claim that this is a literal visualization of the LLM’s internal brain. The goal IS to build a graph-structured external memory layer for an LLM and test whether selective amnesia / selective forgetting improves performance, controllability, and debugging.

CORE RESEARCH QUESTION
Does a graph-structured, human-editable external memory with selective forgetting outperform flat memory and vector retrieval baselines on long-term memory tasks, especially under contradiction, drift, and memory budget pressure?

WHAT I WANT FROM YOU
I want you to design and implement the project scaffold, propose the experiments, and leave the system easy for me to modify. Make strong default choices, but keep everything configurable and hackable.

IMPORTANT STYLE REQUIREMENTS
- Do not overclaim novelty.
- Do not frame this as “mapping the literal LLM brain.”
- Frame it as graph-structured external memory for LLM agents.
- Make reasonable assumptions when needed instead of blocking on questions.
- Prefer simple, readable, research-friendly code over clever abstractions.
- Leave room for me to fiddle with policies, prompts, memory schemas, ranking, and UI.
- Be explicit about what is implemented now vs placeholder vs future work.
- Do not invent benchmark details or fake citations.
- If you reference benchmarks or papers from memory, label them as “to verify.”
- Build a clean MVP first, then a stronger experimental version.

PRIMARY DELIVERABLES
I want you to produce the following, in order:

1. A short architecture summary
2. A repo structure
3. A step-by-step implementation plan
4. The actual code scaffold for the project
5. A config-driven experimental suite
6. A simple visual graph UI
7. Documentation for how to run, modify, and extend it
8. A list of ablations and evaluation metrics
9. A backlog of useful next steps

TECHNICAL GOALS
The system should support:

A. MEMORY REPRESENTATION
- Typed nodes such as:
  - entity
  - fact
  - event
  - user_preference
  - task
  - summary
  - source
  - belief
- Typed edges such as:
  - related_to
  - refers_to
  - supports
  - contradicts
  - supersedes
  - caused_by
  - temporal_before
  - derived_from
  - similar_to

B. MEMORY OPERATIONS
Implement these memory actions as separate, swappable modules:
- add
- update
- merge
- summarize/compress
- decay
- quarantine
- supersede
- delete
- retrieve

C. SELECTIVE AMNESIA / FORGETTING POLICIES
Implement forgetting as modular policies, not hardcoded logic. Include at least:
- recency-based forgetting
- importance-based forgetting
- contradiction-aware supersession
- uncertainty-aware quarantine
- budget-aware pruning
- compression-based forgetting
- hybrid scoring policy

Each policy should be configurable and easy to swap.

D. RETRIEVAL / REASONING MODES
Support at least:
- no external memory
- flat memory list
- vector-style retrieval placeholder interface
- graph retrieval using neighbors / subgraphs / typed edge traversal

E. UI / VISUALIZATION
Build a basic Obsidian-like graph exploration interface for the external memory:
- inspect nodes
- inspect edges
- show metadata
- filter by node/edge type
- highlight contradiction and supersession links
- allow manual deletion or disabling of memories
- allow manual edit of node labels / edge types if feasible
Keep it simple and practical.

F. MODULARITY
Make this project easy to modify via:
- YAML or TOML config files
- policy registry / plugin interfaces
- clearly separated folders
- docstrings and comments
- minimal hidden coupling

RECOMMENDED STACK
Use a practical stack unless you strongly justify something else:
- Python 3.11+
- pydantic for schemas
- networkx for graph structure
- sqlite or lightweight local storage
- optional vector backend interface stub
- Streamlit or Gradio for quick UI
- matplotlib / pyvis / plotly only if needed for graph rendering
- pytest for tests

If you choose a different stack, explain why briefly.

REPO EXPECTATIONS
Create a repo structure like this or better:

- README.md
- pyproject.toml
- requirements.txt or equivalent
- configs/
  - base.yaml
  - experiments/
- src/
  - main app entrypoints
  - memory/
    - schemas
    - graph store
    - flat store
    - retrievers
    - forgetting policies
    - update policies
  - agents/
  - evaluation/
  - ui/
  - utils/
- scripts/
  - run demo
  - generate synthetic data
  - run experiments
- tests/
- notebooks/ or examples/
- docs/

EXPERIMENTAL SUITE
Design the experiments so I can test the core question.

I want baseline conditions such as:
- B0: no external memory
- B1: flat memory
- B2: vector-style retrieval baseline
- B3: graph memory without forgetting
- B4: graph memory with selective forgetting
- B5: graph memory with selective forgetting and human edits enabled if feasible

I want evaluation tasks such as:
- factual recall over multi-session dialogue
- temporal reasoning over events
- contradiction handling
- preference updates over time
- stale memory suppression
- multi-hop retrieval across linked memories
- budget-limited memory retention
- robustness to noisy or irrelevant memory insertion

I also want synthetic stress tests:
- preference changes twice
- facts corrected later
- conflicting memories from different sources
- temporary state mistaken as stable preference
- memory budget reduced at multiple levels
- noisy distractor memories added
- removal of important hub nodes

METRICS
Track at least:
- answer accuracy
- stale retrieval rate
- contradiction error rate
- memory precision
- memory recall
- retrieval latency
- memory growth over time
- token or request cost proxy
- number of useful memories retained
- number of bad memories retained
- effect of forgetting on performance

For graph-specific analysis, include:
- path length to supporting evidence
- correct contradiction links
- correct supersession links
- cluster purity if feasible
- editability/debuggability notes

HUMAN-EDITABILITY ANGLE
I want the project to explicitly preserve room for manual intervention.
Design the system so a human can:
- inspect a bad answer
- trace which memories were retrieved
- see the supporting nodes/edges
- disable, delete, or edit problematic memories
- rerun the query after editing

This is important.

CODING REQUIREMENTS
- Use typed Python where reasonable.
- Use pydantic models for core schemas.
- Keep business logic separated from UI.
- Put important hyperparameters in config.
- Add TODO markers for parts I may want to improve.
- Include small example datasets or synthetic data generators.
- Make the code runnable with minimal setup.
- Avoid giant monolithic files.
- Include tests for the core memory operations and forgetting policies.

DOCUMENTATION REQUIREMENTS
I want:
- setup instructions
- how to run the demo
- how to run experiments
- how to add a new forgetting policy
- how to add a new retriever
- how to change node/edge schemas
- how to inspect and edit the graph
- what assumptions the prototype makes
- current limitations

HOW TO INTERACT WITH ME
Do this in phases. In your first response:
1. Briefly restate the project in clean research terms
2. Propose the architecture
3. Propose the repo structure
4. Propose the experimental plan
5. List key assumptions
6. Then start generating the most important files first

Do not waste time with generic motivational text.
Do not give me a high-level only answer.
Actually start building the project in the response.

PRIORITY ORDER
Prioritize in this order:
1. clean architecture
2. runnable scaffold
3. configurable experiments
4. usable graph UI
5. polish

WHAT I CARE ABOUT MOST
I care most about:
- graph memory abstraction
- selective forgetting policies
- easy experimentation
- easy manual modification
- clear baselines
- no overclaiming about “LLM brain”
- something I can actually run and tinker with

OPTIONAL NICE-TO-HAVES
If practical, add:
- export/import graph as JSON
- replay logs of memory changes
- explainability trace for retrieval
- side-by-side comparison of two policies
- simple dashboard of memory stats over time

PLEASE START NOW
Start with:
- concise project framing
- architecture
- repo tree
- implementation phases
- then generate the first core files

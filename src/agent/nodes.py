import hashlib
from typing import List

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from agent.state import State
from agent.llm import build_llm
from memory.graph_store import MemoryGraph
from memory.schemas import MemoryNode, MemoryEdge, NodeType, EdgeType
from memory.forgetting import ForgetPolicy


# ---------------------------------------------------------------------------
# Extraction schema — lightweight types the LLM fills in
# ---------------------------------------------------------------------------

class ExtractedNode(BaseModel):
    """A single memory node extracted from conversation."""
    content: str = Field(description="A concise description of the entity, fact, event, preference, task, belief, summary, or source.")
    type: str = Field(description="One of: entity, fact, event, user_preference, task, summary, source, belief")

class ExtractedEdge(BaseModel):
    """A relationship between two extracted nodes, referenced by index."""
    source_index: int = Field(description="0-based index of the source node in the nodes list.")
    target_index: int = Field(description="0-based index of the target node in the nodes list.")
    type: str = Field(description="One of: related_to, refers_to, supports, contradicts, supersedes, caused_by, temporal_before, derived_from, similar_to")

class ExtractedGraph(BaseModel):
    """A set of memory nodes and edges extracted from a conversation."""
    nodes: List[ExtractedNode] = Field(default_factory=list)
    edges: List[ExtractedEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a memory graph builder. Given a conversation, extract a detailed \
knowledge graph of memorable information.

Extract nodes of these types:
- entity: A named person, place, organisation, or concrete object.
- fact: A stated piece of knowledge or claim.
- event: Something that happened or will happen.
- user_preference: A preference, like, or dislike expressed by the user.
- task: An action item or goal mentioned by the user.
- belief: An opinion or subjective stance.
- summary: A condensed recap of a longer exchange.
- source: A referenced document, URL, or external resource.

Extract edges of these types:
- related_to: General association.
- refers_to: One node references another.
- supports: One node provides evidence for another.
- contradicts: One node conflicts with another.
- supersedes: One node replaces/updates another.
- caused_by: Causal relationship.
- temporal_before: Temporal ordering.
- derived_from: One node was inferred from another.
- similar_to: Semantic similarity.

Rules:
- Node content MUST be a descriptive sentence that captures context, not just \
a bare name. For example, instead of "Suren", write "Suren is the user \
who is building a memory graph agent". Instead of "Paris", write \
"Paris is the capital of France, mentioned when discussing European cities".
- Always explain WHO, WHAT, or WHY in the node content so the node is \
meaningful on its own without the original conversation.
- Extract relationships (edges) between nodes. A good memory graph has \
both detailed nodes AND edges connecting them.
- Reference edge endpoints by their 0-based index in the nodes list. \
For example, if node 0 is about "Alice" and node 1 is about "Bob", \
an edge from Alice to Bob has source_index=0, target_index=1.
- Only extract information that is worth remembering long-term.
- Do not fabricate information not present in the conversation.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_id(content: str, node_type: str) -> str:
    """Deterministic short hash so the same content always maps to the same id."""
    return hashlib.sha256(f"{node_type}:{content}".encode()).hexdigest()[:12]


def _edge_id(source_id: str, target_id: str, edge_type: str) -> str:
    return hashlib.sha256(f"{source_id}-{edge_type}-{target_id}".encode()).hexdigest()[:12]


_NODE_TYPE_MAP = {t.value: t for t in NodeType}
_EDGE_TYPE_MAP = {t.value: t for t in EdgeType}


def _messages_to_text(messages: List[BaseMessage]) -> str:
    """Render messages into a readable transcript for the extraction prompt."""
    lines = []
    for msg in messages:
        role = msg.type  # "human", "ai", "system"
        lines.append(f"[{role}]: {msg.content}")
    return "\n".join(lines)


def _convert_extracted(extracted: ExtractedGraph, existing_ids: set[str]) -> MemoryGraph:
    """Convert LLM output into proper MemoryNode / MemoryEdge objects,
    skipping any nodes whose id already exists in the graph."""
    nodes: list[MemoryNode] = []
    # Map each extracted index → deterministic node id (even for dupes)
    index_to_id: dict[int, str] = {}

    for i, en in enumerate(extracted.nodes):
        ntype = _NODE_TYPE_MAP.get(en.type)
        if ntype is None:
            continue
        nid = _node_id(en.content, en.type)
        index_to_id[i] = nid
        if nid in existing_ids:
            continue
        nodes.append(MemoryNode(id=nid, content=en.content, type=ntype))

    edges: list[MemoryEdge] = []
    for ee in extracted.edges:
        src = index_to_id.get(ee.source_index)
        tgt = index_to_id.get(ee.target_index)
        etype = _EDGE_TYPE_MAP.get(ee.type)
        if src is None or tgt is None or etype is None or src == tgt:
            continue
        eid = _edge_id(src, tgt, ee.type)
        edges.append(MemoryEdge(id=eid, source=src, target=tgt, type=etype))

    return MemoryGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Module-level forget policy (accumulates access metadata across calls)
# ---------------------------------------------------------------------------

_forget_policy = ForgetPolicy()


# ---------------------------------------------------------------------------
# Agent graph nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: State) -> dict:
    """Retrieve relevant memory nodes and inject them as context for generation."""
    memory = state["memory"]
    if not memory.nodes:
        return {"messages": []}

    # Find the latest user query.
    query = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            query = msg.content.lower()
            break

    if not query:
        return {"messages": []}

    # Score nodes by keyword overlap with the query.
    query_words = set(query.split())
    scored = []
    for node in memory.nodes:
        node_words = set(node.content.lower().split())
        overlap = len(query_words & node_words)
        scored.append((overlap, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    relevant = [node for score, node in scored if score > 0][:10]

    if not relevant:
        # No keyword match — fall back to the first few nodes.
        relevant = memory.nodes[:10]

    # Track accessed nodes so the forget policy knows what was used.
    _forget_policy.track_many(relevant)

    lines = [f"- [{n.type.value}] {n.content}" for n in relevant]
    context_msg = SystemMessage(content="Retrieved memories:\n" + "\n".join(lines))
    return {"messages": [context_msg]}


def generate_node(state: State) -> dict:
    """Generate a response using the LLM with memory-augmented messages."""
    llm = build_llm(model="llama3.2:latest", provider="ollama")

    # Reorder: system messages first, then conversation in original order.
    # retrieve_node appends a SystemMessage via operator.add, which lands
    # after the last HumanMessage — confusing the LLM.  Fix by grouping.
    system_msgs = [m for m in state["messages"] if m.type == "system"]
    conversation = [m for m in state["messages"] if m.type != "system"]
    messages = system_msgs + conversation

    response = llm.invoke(messages)
    return {"messages": [response]}


def forget_node(state: State) -> dict:
    """Apply the forgetting policy to prune low-value memories."""
    pruned = _forget_policy.apply(state["memory"])
    return {"memory": pruned}


# ---------------------------------------------------------------------------
# Graph store builder — runs AFTER generate_node so it can extract
# knowledge from the latest human + AI exchange.
# ---------------------------------------------------------------------------

def build_graph_store(state: State) -> dict:
    """Extract new knowledge from the latest exchange into the memory graph.

    Finds the most recent human → AI turn, runs structured LLM extraction,
    and merges the resulting nodes/edges into the existing graph
    (deduplicated by deterministic content-hash ids).
    """
    messages = state["messages"]

    # Locate the most recent human → AI exchange.
    last_human = None
    last_ai = None
    for msg in reversed(messages):
        if msg.type == "ai" and last_ai is None:
            last_ai = msg
        elif msg.type == "human" and last_human is None:
            last_human = msg
        if last_human and last_ai:
            break

    if not last_human or not last_ai:
        return {}

    llm = build_llm(model="llama3.2:latest", provider="ollama")
    structured_llm = llm.with_structured_output(ExtractedGraph)

    conversation_text = _messages_to_text([last_human, last_ai])
    extract_messages = [
        SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Extract a memory graph from this conversation:\n\n{conversation_text}"
        ),
    ]

    extracted: ExtractedGraph = structured_llm.invoke(extract_messages)

    # Merge new extractions into the existing graph.
    existing = state["memory"]
    existing_node_ids = {n.id for n in existing.nodes}
    existing_edge_ids = {e.id for e in existing.edges}

    new_graph = _convert_extracted(extracted, existing_node_ids)

    # Track newly created nodes.
    _forget_policy.track_many(new_graph.nodes)

    merged_nodes = list(existing.nodes) + new_graph.nodes
    merged_edges = list(existing.edges) + [
        e for e in new_graph.edges if e.id not in existing_edge_ids
    ]

    return {"memory": MemoryGraph(nodes=merged_nodes, edges=merged_edges)}
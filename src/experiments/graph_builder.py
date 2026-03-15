"""Build MemoryGraph from benchmark conversations.

Processes dialogue sessions incrementally, extracts nodes/edges via the LLM,
links cross-session entities through fuzzy deduplication, and optionally
applies the ForgetPolicy after each session.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import CACHE_DIR, DEFAULT_MODEL, DEFAULT_PROVIDER, EXTRACTION_BATCH_TURNS
from experiments.data_loaders import (
    LoCoMoSample,
    LongMemInstance,
    locomo_session_to_text,
    longmem_session_to_text,
)
from memory.graph_store import MemoryGraph
from memory.schemas import MemoryNode, MemoryEdge, NodeType, EdgeType
from memory.forgetting import ForgetPolicy
from agent.llm import build_llm
from agent.nodes import (
    ExtractedGraph,
    _EXTRACTION_SYSTEM_PROMPT,
    _node_id,
    _edge_id,
    _NODE_TYPE_MAP,
    _EDGE_TYPE_MAP,
)


# =====================================================================
# Stopwords for content similarity
# =====================================================================

_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "between", "out", "off", "over", "under",
    "about", "up", "down", "and", "but", "or", "if", "so", "than",
    "that", "this", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "who", "what", "which", "when", "where",
    "how", "not", "no", "also", "just", "very", "more", "some",
}


def _content_words(text: str) -> Set[str]:
    """Extract meaningful words from text (lowercased, stopwords removed)."""
    return set(text.lower().split()) - _STOPWORDS


def _content_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity between the content words of two texts."""
    words_a = _content_words(text_a)
    words_b = _content_words(text_b)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


# =====================================================================
# Cache helpers
# =====================================================================

def _cache_key(prefix: str, sample_id: str, forget_preset: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{prefix}_{sample_id}_{forget_preset}.json"


def _save_graph(path: Path, graph: MemoryGraph) -> None:
    path.write_text(graph.model_dump_json(indent=2))


def _load_graph(path: Path) -> Optional[MemoryGraph]:
    if path.exists():
        return MemoryGraph.model_validate_json(path.read_text())
    return None


# =====================================================================
# Context-aware extraction with cross-session edge formation
# =====================================================================

_DEDUP_THRESHOLD = 0.70   # Jaccard above this → treat as same node (redirect)
_RELATED_THRESHOLD = 0.35  # Jaccard above this → create a related_to edge


def _build_existing_context(existing: MemoryGraph, max_nodes: int = 30) -> str:
    """Summarise existing nodes so the LLM knows what's already in the graph."""
    if not existing.nodes:
        return ""
    # Prioritise entities and facts — they're most likely to recur
    priority = {NodeType.ENTITY: 0, NodeType.FACT: 1, NodeType.USER_PREFERENCE: 2}
    sorted_nodes = sorted(existing.nodes, key=lambda n: priority.get(n.type, 9))
    lines = ["Already stored in memory (do not re-extract these, but you may "
             "reference them when creating edges):"]
    for node in sorted_nodes[:max_nodes]:
        lines.append(f"  - [{node.type.value}] {node.content}")
    return "\n".join(lines)


def extract_and_merge(
    text: str,
    existing: MemoryGraph,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    llm=None,
) -> Tuple[MemoryGraph, List[MemoryNode]]:
    """Run structured extraction on *text* and merge into *existing*.

    Returns ``(merged_graph, new_nodes)`` so callers can track only the
    newly created nodes.
    """
    if llm is None:
        llm = build_llm(model=model, provider=provider)

    structured_llm = llm.with_structured_output(ExtractedGraph)

    # Include existing node context so the LLM can avoid duplicates
    existing_context = _build_existing_context(existing)
    prompt_parts = ["Extract a memory graph from this conversation:\n"]
    if existing_context:
        prompt_parts.append(existing_context + "\n")
    prompt_parts.append(text)

    messages = [
        SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(prompt_parts)),
    ]

    try:
        extracted: ExtractedGraph = structured_llm.invoke(messages)
    except Exception as e:
        print(f"  [extraction error: {e}]")
        return existing, []

    # ------------------------------------------------------------------
    # Convert extracted nodes, applying fuzzy dedup against existing
    # ------------------------------------------------------------------
    existing_node_ids = {n.id for n in existing.nodes}
    existing_edge_ids = {e.id for e in existing.edges}

    new_nodes: List[MemoryNode] = []
    index_to_id: Dict[int, str] = {}
    cross_session_edges: List[MemoryEdge] = []

    for i, en in enumerate(extracted.nodes):
        ntype = _NODE_TYPE_MAP.get(en.type)
        if ntype is None:
            continue

        nid = _node_id(en.content, en.type)
        index_to_id[i] = nid

        # Exact hash dedup
        if nid in existing_node_ids:
            continue

        # Fuzzy dedup — check if a very similar node already exists
        best_sim = 0.0
        best_existing_id: Optional[str] = None
        for enode in existing.nodes:
            sim = _content_similarity(en.content, enode.content)
            if sim > best_sim:
                best_sim = sim
                best_existing_id = enode.id

        if best_sim >= _DEDUP_THRESHOLD and best_existing_id is not None:
            # Redirect this index to the existing node
            index_to_id[i] = best_existing_id
        else:
            # Truly new node
            new_nodes.append(MemoryNode(id=nid, content=en.content, type=ntype))
            # If moderately similar to an existing node, create a cross-session edge
            if best_sim >= _RELATED_THRESHOLD and best_existing_id is not None:
                eid = _edge_id(nid, best_existing_id, "related_to")
                if eid not in existing_edge_ids:
                    cross_session_edges.append(MemoryEdge(
                        id=eid, source=nid, target=best_existing_id,
                        type=EdgeType.RELATED_TO,
                    ))

    # ------------------------------------------------------------------
    # Convert extracted edges (index-based → id-based)
    # ------------------------------------------------------------------
    new_edges: List[MemoryEdge] = []
    for ee in extracted.edges:
        src = index_to_id.get(ee.source_index)
        tgt = index_to_id.get(ee.target_index)
        etype = _EDGE_TYPE_MAP.get(ee.type)
        if src is None or tgt is None or etype is None or src == tgt:
            continue
        eid = _edge_id(src, tgt, ee.type)
        if eid not in existing_edge_ids:
            new_edges.append(MemoryEdge(id=eid, source=src, target=tgt, type=etype))

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    merged_nodes = list(existing.nodes) + new_nodes
    merged_edges = (
        list(existing.edges)
        + [e for e in new_edges if e.id not in existing_edge_ids]
        + cross_session_edges
    )

    return MemoryGraph(nodes=merged_nodes, edges=merged_edges), new_nodes


# =====================================================================
# LoCoMo: build graph session-by-session
# =====================================================================

def build_locomo_graph(
    sample: LoCoMoSample,
    *,
    forget_preset: str = "none",
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    use_cache: bool = True,
) -> MemoryGraph:
    cache_path = _cache_key("locomo", sample.sample_id, forget_preset)
    if use_cache:
        cached = _load_graph(cache_path)
        if cached is not None:
            print(f"  [cache hit: {cache_path.name}]")
            return cached

    from experiments.config import FORGET_PRESETS
    policy = ForgetPolicy(**FORGET_PRESETS[forget_preset])
    graph = MemoryGraph(nodes=[], edges=[])
    llm = build_llm(model=model, provider=provider)

    for sess in sample.sessions:
        text = locomo_session_to_text(sess, sample.speaker_a, sample.speaker_b)
        graph, new_nodes = extract_and_merge(text, graph, model=model, provider=provider, llm=llm)

        # Only register newly extracted nodes (not all nodes)
        for node in new_nodes:
            policy.register_new(node)

        if forget_preset != "none":
            graph = policy.apply(graph)

        print(f"  session {sess.session_num}: {len(graph.nodes)} nodes, {len(graph.edges)} edges (+{len(new_nodes)} new)")

    _save_graph(cache_path, graph)
    return graph


# =====================================================================
# LongMemEval: build graph from haystack sessions
# =====================================================================

def build_longmemeval_graph(
    instance: LongMemInstance,
    *,
    forget_preset: str = "none",
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    use_cache: bool = True,
) -> MemoryGraph:
    cache_path = _cache_key("longmem", instance.question_id, forget_preset)
    if use_cache:
        cached = _load_graph(cache_path)
        if cached is not None:
            return cached

    from experiments.config import FORGET_PRESETS
    policy = ForgetPolicy(**FORGET_PRESETS[forget_preset])
    graph = MemoryGraph(nodes=[], edges=[])
    llm = build_llm(model=model, provider=provider)

    for sess in instance.sessions:
        text = longmem_session_to_text(sess)
        graph, new_nodes = extract_and_merge(text, graph, model=model, provider=provider, llm=llm)

        # Only register newly extracted nodes
        for node in new_nodes:
            policy.register_new(node)

        if forget_preset != "none":
            graph = policy.apply(graph)

    _save_graph(cache_path, graph)
    return graph

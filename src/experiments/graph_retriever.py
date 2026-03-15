"""Retrieve relevant context from a MemoryGraph for a given query.

Uses stopword-filtered keyword overlap scoring with normalisation,
plus graph-neighbour expansion for relational context.
"""
from __future__ import annotations

from collections import Counter
from math import log
from typing import Dict, List, Set, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.graph_store import MemoryGraph
from memory.schemas import MemoryEdge, MemoryNode

# =====================================================================
# Stopwords — filtered from both query and node text before scoring
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


def _clean_words(text: str) -> Set[str]:
    """Lowercase, split, remove stopwords."""
    return set(text.lower().split()) - _STOPWORDS


def _keyword_score(query_words: Set[str], text: str) -> float:
    """Normalised keyword overlap: fraction of query terms found in text."""
    text_words = _clean_words(text)
    if not query_words:
        return 0.0
    overlap = len(query_words & text_words)
    return overlap / len(query_words)


def _idf_score(
    query_words: Set[str],
    text: str,
    doc_freq: Dict[str, int],
    n_docs: int,
) -> float:
    """TF-IDF–like score: matched query terms weighted by inverse doc frequency."""
    text_words = _clean_words(text)
    if not query_words or n_docs == 0:
        return 0.0
    score = 0.0
    for w in query_words & text_words:
        df = doc_freq.get(w, 0)
        idf = log((n_docs + 1) / (df + 1)) + 1.0  # smoothed IDF
        score += idf
    return score


def _build_doc_freq(nodes: List[MemoryNode]) -> Dict[str, int]:
    """Count how many nodes contain each word (for IDF weighting)."""
    df: Dict[str, int] = Counter()
    for node in nodes:
        words = _clean_words(node.content)
        for w in words:
            df[w] += 1
    return df


def _neighbour_ids(node_id: str, edges: List[MemoryEdge]) -> Set[str]:
    """Return IDs of nodes directly connected to *node_id*."""
    result: Set[str] = set()
    for e in edges:
        if e.source == node_id:
            result.add(e.target)
        elif e.target == node_id:
            result.add(e.source)
    return result


def retrieve_nodes(
    query: str,
    graph: MemoryGraph,
    top_k: int = 10,
    expand_neighbours: bool = True,
) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
    """Return the most relevant nodes (and connecting edges) for *query*.

    1. Score every node by IDF-weighted keyword overlap with the query.
    2. Take the top-k scoring nodes.
    3. Optionally expand: for each top-k node, include its 1-hop neighbours
       (up to 2x top_k total).
    4. Return all edges whose source AND target are in the returned node set.
    """
    if not graph.nodes:
        return [], []

    query_words = _clean_words(query)
    if not query_words:
        # Fallback: return first few nodes
        return graph.nodes[:top_k], []

    # Build IDF from the full graph
    doc_freq = _build_doc_freq(graph.nodes)
    n_docs = len(graph.nodes)

    scored: List[Tuple[float, MemoryNode]] = []
    for node in graph.nodes:
        score = _idf_score(query_words, node.content, doc_freq, n_docs)
        scored.append((score, node))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top-k (include nodes with score 0 if we don't have enough matches)
    top_nodes = [n for _, n in scored[:top_k]]
    selected_ids: Set[str] = {n.id for n in top_nodes}

    # Expand with 1-hop neighbours
    if expand_neighbours:
        for node in list(top_nodes):
            for nid in _neighbour_ids(node.id, graph.edges):
                selected_ids.add(nid)
        # Cap at 2x top_k
        if len(selected_ids) > 2 * top_k:
            extra_ids = selected_ids - {n.id for n in top_nodes}
            id_to_node = {n.id: n for n in graph.nodes}
            extra_scored = [
                (_idf_score(query_words, id_to_node[nid].content, doc_freq, n_docs), nid)
                for nid in extra_ids if nid in id_to_node
            ]
            extra_scored.sort(key=lambda x: x[0], reverse=True)
            keep_extra = {nid for _, nid in extra_scored[:top_k]}
            selected_ids = {n.id for n in top_nodes} | keep_extra

    id_to_node = {n.id: n for n in graph.nodes}
    result_nodes = [id_to_node[nid] for nid in selected_ids if nid in id_to_node]

    result_edges = [
        e for e in graph.edges
        if e.source in selected_ids and e.target in selected_ids
    ]

    return result_nodes, result_edges


def format_context(
    nodes: List[MemoryNode],
    edges: List[MemoryEdge],
    max_edges: int = 15,
) -> str:
    """Format retrieved nodes and edges as a text block for the LLM prompt.

    Limits edges to *max_edges* to avoid overwhelming the LLM with context.
    Prioritises structural edges (supersedes, contradicts, temporal_before)
    over generic related_to edges.
    """
    if not nodes:
        return "(no relevant memories)"

    id_to_content: Dict[str, str] = {n.id: n.content for n in nodes}
    lines: List[str] = []

    lines.append("Memories:")
    for n in nodes:
        lines.append(f"  - [{n.type.value}] {n.content}")

    if edges:
        # Prioritise informative edge types
        _EDGE_PRIORITY = {
            "supersedes": 0, "contradicts": 1, "temporal_before": 2,
            "caused_by": 3, "supports": 4, "derived_from": 5,
            "refers_to": 6, "similar_to": 7, "related_to": 8,
        }
        sorted_edges = sorted(edges, key=lambda e: _EDGE_PRIORITY.get(e.type.value, 9))
        shown = sorted_edges[:max_edges]

        lines.append("Relationships:")
        for e in shown:
            src = id_to_content.get(e.source, "?")
            tgt = id_to_content.get(e.target, "?")
            lines.append(f'  - "{src}" --[{e.type.value}]--> "{tgt}"')

    return "\n".join(lines)


def retrieve_and_format(
    query: str,
    graph: MemoryGraph,
    top_k: int = 10,
) -> str:
    """Convenience: retrieve + format in one call."""
    nodes, edges = retrieve_nodes(query, graph, top_k=top_k)
    return format_context(nodes, edges)

"""Retrieve relevant context from a MemoryGraph for a given query.

Uses keyword overlap scoring + graph-neighbour expansion.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.graph_store import MemoryGraph
from memory.schemas import MemoryEdge, MemoryNode


def _keyword_score(query_words: Set[str], text: str) -> int:
    return len(query_words & set(text.lower().split()))


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

    1. Score every node by keyword overlap with the query.
    2. Take the top-k scoring nodes.
    3. Optionally expand: for each top-k node, include its 1-hop neighbours
       (up to 2× top_k total).
    4. Return all edges whose source AND target are in the returned node set.
    """
    if not graph.nodes:
        return [], []

    query_words = set(query.lower().split())

    scored: List[Tuple[int, MemoryNode]] = []
    for node in graph.nodes:
        score = _keyword_score(query_words, node.content)
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
        # Cap at 2× top_k
        if len(selected_ids) > 2 * top_k:
            # Keep the original top-k plus highest-scoring neighbours
            extra_ids = selected_ids - {n.id for n in top_nodes}
            id_to_node = {n.id: n for n in graph.nodes}
            extra_scored = [
                (_keyword_score(query_words, id_to_node[nid].content), nid)
                for nid in extra_ids if nid in id_to_node
            ]
            extra_scored.sort(key=lambda x: x[0], reverse=True)
            keep_extra = {nid for _, nid in extra_scored[: top_k]}
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
) -> str:
    """Format retrieved nodes and edges as a text block for the LLM prompt."""
    if not nodes:
        return "(no relevant memories)"

    id_to_content: Dict[str, str] = {n.id: n.content for n in nodes}
    lines: List[str] = []

    lines.append("Memories:")
    for n in nodes:
        lines.append(f"  - [{n.type.value}] {n.content}")

    if edges:
        lines.append("Relationships:")
        for e in edges:
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

"""Memory retrieval strategies: flat list, graph-based with neighbor traversal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.memory.graph_store import GraphStore
from src.memory.schemas import MemoryNode, RetrievedMemory


class MemoryRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        graph_store: GraphStore,
        max_results: int,
        current_turn: int,
    ) -> list[RetrievedMemory]:
        ...


class NoneRetriever(MemoryRetriever):
    """No external memory. Baseline B0."""

    def retrieve(
        self, query: str, graph_store: GraphStore, max_results: int, current_turn: int
    ) -> list[RetrievedMemory]:
        return []


class FlatRetriever(MemoryRetriever):
    """Return all memories sorted by recency. Baseline B1."""

    def retrieve(
        self, query: str, graph_store: GraphStore, max_results: int, current_turn: int
    ) -> list[RetrievedMemory]:
        nodes = graph_store.get_all_nodes()
        nodes.sort(key=lambda n: n.last_accessed, reverse=True)
        return [
            RetrievedMemory(node=n, relevance_score=1.0 / (i + 1))
            for i, n in enumerate(nodes[:max_results])
        ]


class GraphRetriever(MemoryRetriever):
    """Graph-aware retrieval with keyword scoring and neighbor expansion."""

    def __init__(self, neighbor_depth: int = 1, include_edge_context: bool = True):
        self.neighbor_depth = neighbor_depth
        self.include_edge_context = include_edge_context

    def retrieve(
        self, query: str, graph_store: GraphStore, max_results: int, current_turn: int
    ) -> list[RetrievedMemory]:
        nodes = graph_store.get_all_nodes()
        if not nodes:
            return []

        # Score all nodes by keyword relevance
        scored: list[tuple[MemoryNode, float]] = []
        for node in nodes:
            score = self._relevance_score(query, node.content)
            if score > 0:
                scored.append((node, score))

        # If no keyword matches, fall back to recency
        if not scored:
            nodes.sort(key=lambda n: n.last_accessed, reverse=True)
            scored = [(n, 0.1) for n in nodes[:max_results]]

        # Sort by relevance, take top candidates
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: max_results * 2]  # over-fetch for re-ranking

        # Expand with graph neighbors
        results: list[RetrievedMemory] = []
        for node, score in top:
            neighbors = graph_store.get_neighbors(
                node.id, depth=self.neighbor_depth
            )
            edges = graph_store.get_edges(node.id) if self.include_edge_context else []
            # Boost score for well-connected nodes
            boosted = score + 0.05 * len(neighbors)
            results.append(
                RetrievedMemory(
                    node=node,
                    edges=edges,
                    neighbors=neighbors,
                    relevance_score=boosted,
                )
            )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:max_results]

    @staticmethod
    def _relevance_score(query: str, content: str) -> float:
        """Token-level Jaccard similarity. TODO: replace with embedding cosine similarity."""
        q_tokens = set(query.lower().split())
        c_tokens = set(content.lower().split())
        if not q_tokens or not c_tokens:
            return 0.0
        intersection = q_tokens & c_tokens
        union = q_tokens | c_tokens
        return len(intersection) / len(union)


# ── Registry ─────────────────────────────────────────────────────

RETRIEVER_REGISTRY: dict[str, type[MemoryRetriever]] = {
    "none": NoneRetriever,
    "flat": FlatRetriever,
    "graph": GraphRetriever,
}


def get_retriever(name: str, config: dict[str, Any]) -> MemoryRetriever:
    cls = RETRIEVER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown retriever: {name!r}. Available: {list(RETRIEVER_REGISTRY)}")

    ret_cfg = config.get("retrieval", {})
    if name == "graph":
        return GraphRetriever(
            neighbor_depth=ret_cfg.get("neighbor_depth", 1),
            include_edge_context=ret_cfg.get("include_edge_context", True),
        )
    return cls()

"""NetworkX-based graph store for memory nodes and edges."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from src.memory.schemas import EdgeType, MemoryEdge, MemoryNode


class GraphStore:
    """Wraps a networkx DiGraph with typed CRUD operations for MemoryNode/MemoryEdge."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._nodes: dict[str, MemoryNode] = {}

    # ── CRUD: Nodes ──────────────────────────────────────────────

    def add_node(self, node: MemoryNode) -> str:
        self._nodes[node.id] = node
        self._graph.add_node(node.id)
        return node.id

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def update_node(self, node_id: str, **kwargs: Any) -> MemoryNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id} not found")
        for k, v in kwargs.items():
            setattr(node, k, v)
        return node

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        self._graph.remove_node(node_id)
        return True

    def disable_node(self, node_id: str) -> None:
        """Soft-delete / quarantine a node."""
        if node_id in self._nodes:
            self._nodes[node_id].enabled = False

    def enable_node(self, node_id: str) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].enabled = True

    # ── CRUD: Edges ──────────────────────────────────────────────

    def add_edge(self, edge: MemoryEdge) -> None:
        if edge.source_id not in self._nodes:
            raise KeyError(f"Source node {edge.source_id} not found")
        if edge.target_id not in self._nodes:
            raise KeyError(f"Target node {edge.target_id} not found")
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            created_at=edge.created_at.isoformat(),
            metadata=edge.metadata,
        )

    def get_edges(
        self, node_id: str, direction: str = "both"
    ) -> list[MemoryEdge]:
        edges: list[MemoryEdge] = []
        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(node_id, data=True):
                edges.append(
                    MemoryEdge(
                        source_id=node_id,
                        target_id=target,
                        edge_type=EdgeType(data.get("edge_type", "related_to")),
                        weight=data.get("weight", 1.0),
                        metadata=data.get("metadata", {}),
                    )
                )
        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(node_id, data=True):
                edges.append(
                    MemoryEdge(
                        source_id=source,
                        target_id=node_id,
                        edge_type=EdgeType(data.get("edge_type", "related_to")),
                        weight=data.get("weight", 1.0),
                        metadata=data.get("metadata", {}),
                    )
                )
        return edges

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        if self._graph.has_edge(source_id, target_id):
            self._graph.remove_edge(source_id, target_id)
            return True
        return False

    # ── Queries ──────────────────────────────────────────────────

    def get_all_nodes(self, include_disabled: bool = False) -> list[MemoryNode]:
        nodes = list(self._nodes.values())
        if not include_disabled:
            nodes = [n for n in nodes if n.enabled]
        return nodes

    def get_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        edge_types: list[EdgeType] | None = None,
    ) -> list[MemoryNode]:
        """BFS traversal up to `depth` hops, optionally filtering by edge type."""
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        result: list[MemoryNode] = []

        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue
            # Check both directions
            for neighbor in set(self._graph.successors(current)) | set(
                self._graph.predecessors(current)
            ):
                if neighbor in visited:
                    continue
                # Filter by edge type if specified
                if edge_types is not None:
                    edge_data = self._graph.get_edge_data(current, neighbor) or {}
                    rev_data = self._graph.get_edge_data(neighbor, current) or {}
                    et_fwd = edge_data.get("edge_type")
                    et_rev = rev_data.get("edge_type")
                    allowed = {e.value for e in edge_types}
                    if et_fwd not in allowed and et_rev not in allowed:
                        continue
                visited.add(neighbor)
                node = self._nodes.get(neighbor)
                if node and node.enabled:
                    result.append(node)
                    queue.append((neighbor, d + 1))

        return result

    def search_by_content(self, query: str) -> list[MemoryNode]:
        """Simple substring search. TODO: replace with embedding similarity."""
        query_lower = query.lower()
        return [
            n
            for n in self._nodes.values()
            if n.enabled and query_lower in n.content.lower()
        ]

    def node_count(self, include_disabled: bool = False) -> int:
        if include_disabled:
            return len(self._nodes)
        return sum(1 for n in self._nodes.values() if n.enabled)

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {
                nid: node.model_dump(mode="json") for nid, node in self._nodes.items()
            },
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **data,
                }
                for u, v, data in self._graph.edges(data=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphStore:
        store = cls()
        for nid, node_data in data.get("nodes", {}).items():
            node = MemoryNode(**node_data)
            store._nodes[nid] = node
            store._graph.add_node(nid)
        for edge_data in data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            store._graph.add_edge(source, target, **edge_data)
        return store

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> GraphStore:
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

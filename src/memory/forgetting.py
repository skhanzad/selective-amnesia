from __future__ import annotations

import math
import time
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from memory.schemas import EdgeType, MemoryEdge, MemoryNode, NodeType


# ---------------------------------------------------------------------------
# Node-level metadata (maintained by the policy across apply() calls)
# ---------------------------------------------------------------------------

class NodeMeta(BaseModel):
    """Tracks per-node signals that inform forgetting decisions."""
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 1
    pinned: bool = False  # pinned nodes are never forgotten

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1


# ---------------------------------------------------------------------------
# Forgetting strategy enum
# ---------------------------------------------------------------------------

class ForgetStrategy(str, Enum):
    """Individual scoring dimensions that can be mixed."""
    TEMPORAL_DECAY = "temporal_decay"
    ACCESS_FREQUENCY = "access_frequency"
    STRUCTURAL_IMPORTANCE = "structural_importance"
    TYPE_PRIORITY = "type_priority"
    REDUNDANCY = "redundancy"


# ---------------------------------------------------------------------------
# Default weights / constants
# ---------------------------------------------------------------------------

_DEFAULT_STRATEGY_WEIGHTS: Dict[ForgetStrategy, float] = {
    ForgetStrategy.TEMPORAL_DECAY: 0.30,
    ForgetStrategy.ACCESS_FREQUENCY: 0.20,
    ForgetStrategy.STRUCTURAL_IMPORTANCE: 0.20,
    ForgetStrategy.TYPE_PRIORITY: 0.15,
    ForgetStrategy.REDUNDANCY: 0.15,
}

_DEFAULT_TYPE_PRIORITY: Dict[NodeType, float] = {
    NodeType.USER_PREFERENCE: 1.0,
    NodeType.BELIEF: 0.9,
    NodeType.ENTITY: 0.8,
    NodeType.FACT: 0.75,
    NodeType.TASK: 0.6,
    NodeType.EVENT: 0.5,
    NodeType.SUMMARY: 0.4,
    NodeType.SOURCE: 0.3,
}

# Edges that signal the *target* is outdated / weakened.
_SUPPRESSIVE_EDGE_TYPES: Set[EdgeType] = {
    EdgeType.SUPERSEDES,
    EdgeType.CONTRADICTS,
}


# ---------------------------------------------------------------------------
# ForgetPolicy
# ---------------------------------------------------------------------------

class ForgetPolicy(BaseModel):
    """Selectively prunes nodes and edges from a MemoryGraph.

    Assigns every node a *retention score* in [0, 1] by combining several
    weighted strategy dimensions.  Nodes scoring below ``threshold`` are
    forgotten, and any edges left dangling after node removal are cleaned up.
    Edges can also be independently pruned when both endpoints have low scores.
    """

    # -- tunables --
    threshold: float = Field(
        default=0.35,
        description="Nodes with a retention score below this value are forgotten.",
    )
    max_nodes: Optional[int] = Field(
        default=None,
        description="Hard cap on nodes kept.  If set, the lowest-scoring nodes "
        "are pruned until the graph fits.  Applied after threshold pruning.",
    )
    half_life_seconds: float = Field(
        default=86_400.0,
        description="Half-life for temporal decay (default 1 day).",
    )
    edge_score_threshold: float = Field(
        default=0.25,
        description="Independently prune an edge when *both* endpoints score "
        "below this value (before node pruning).",
    )
    strategy_weights: Dict[ForgetStrategy, float] = Field(
        default_factory=lambda: dict(_DEFAULT_STRATEGY_WEIGHTS),
    )
    type_priority: Dict[NodeType, float] = Field(
        default_factory=lambda: dict(_DEFAULT_TYPE_PRIORITY),
    )

    # -- internal bookkeeping (persisted across calls) --
    _meta: Dict[str, NodeMeta] = {}

    # -- public helpers for metadata management --

    def track(self, node: MemoryNode) -> None:
        """Register a new node or bump its access counters."""
        if node.id in self._meta:
            self._meta[node.id].touch()
        else:
            self._meta[node.id] = NodeMeta()

    def track_many(self, nodes: List[MemoryNode]) -> None:
        for n in nodes:
            self.track(n)

    def pin(self, node_id: str) -> None:
        """Pin a node so it is never forgotten."""
        if node_id in self._meta:
            self._meta[node_id].pinned = True

    def unpin(self, node_id: str) -> None:
        if node_id in self._meta:
            self._meta[node_id].pinned = False

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _temporal_score(self, meta: NodeMeta, now: float) -> float:
        """Exponential decay based on time since last access."""
        age = now - meta.last_accessed
        return math.exp(-math.log(2) * age / self.half_life_seconds)

    def _access_score(self, meta: NodeMeta, max_access: int) -> float:
        """Log-scaled access frequency normalised against the most-accessed node."""
        if max_access <= 1:
            return 1.0
        return math.log1p(meta.access_count) / math.log1p(max_access)

    def _structural_score(self, node_id: str, degree: Dict[str, int], max_degree: int) -> float:
        """Degree centrality: well-connected nodes are more important."""
        if max_degree == 0:
            return 0.5
        return degree.get(node_id, 0) / max_degree

    def _type_score(self, node_type: NodeType) -> float:
        return self.type_priority.get(node_type, 0.5)

    def _redundancy_score(self, node_id: str, suppressed: Set[str]) -> float:
        """Nodes that have been superseded or contradicted score lower."""
        return 0.0 if node_id in suppressed else 1.0

    def score_node(
        self,
        node: MemoryNode,
        *,
        now: float,
        max_access: int,
        degree: Dict[str, int],
        max_degree: int,
        suppressed: Set[str],
    ) -> float:
        meta = self._meta.get(node.id)
        if meta is None:
            # Unknown node — treat as brand-new (full retention).
            return 1.0
        if meta.pinned:
            return 1.0

        scores = {
            ForgetStrategy.TEMPORAL_DECAY: self._temporal_score(meta, now),
            ForgetStrategy.ACCESS_FREQUENCY: self._access_score(meta, max_access),
            ForgetStrategy.STRUCTURAL_IMPORTANCE: self._structural_score(
                node.id, degree, max_degree
            ),
            ForgetStrategy.TYPE_PRIORITY: self._type_score(node.type),
            ForgetStrategy.REDUNDANCY: self._redundancy_score(node.id, suppressed),
        }

        total_weight = sum(self.strategy_weights.get(s, 0.0) for s in scores)
        if total_weight == 0:
            return 1.0
        return sum(
            self.strategy_weights.get(s, 0.0) * v for s, v in scores.items()
        ) / total_weight

    # ------------------------------------------------------------------
    # Graph-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_degree(edges: List[MemoryEdge]) -> Dict[str, int]:
        deg: Dict[str, int] = {}
        for e in edges:
            deg[e.source] = deg.get(e.source, 0) + 1
            deg[e.target] = deg.get(e.target, 0) + 1
        return deg

    @staticmethod
    def _find_suppressed(edges: List[MemoryEdge]) -> Set[str]:
        """Return node ids that are targets of SUPERSEDES / CONTRADICTS edges."""
        return {e.target for e in edges if e.type in _SUPPRESSIVE_EDGE_TYPES}

    # ------------------------------------------------------------------
    # Core: apply the policy
    # ------------------------------------------------------------------

    def apply(self, graph: "MemoryGraph") -> "MemoryGraph":  # noqa: F821
        """Return a new MemoryGraph with low-value nodes and edges removed.

        Steps:
        1. Score every node.
        2. Independently prune weak edges (both endpoints below edge threshold).
        3. Remove nodes below the retention threshold.
        4. If ``max_nodes`` is set, keep only the top-scoring nodes.
        5. Remove dangling edges.
        6. Clean up internal metadata for forgotten nodes.
        """
        from memory.graph_store import MemoryGraph

        if not graph.nodes:
            return MemoryGraph(nodes=[], edges=[])

        now = time.time()

        # Precompute graph-level stats.
        degree = self._compute_degree(graph.edges)
        max_degree = max(degree.values()) if degree else 0
        suppressed = self._find_suppressed(graph.edges)
        max_access = max(
            (m.access_count for m in self._meta.values()), default=1
        )

        # 1. Score nodes.
        scores: Dict[str, float] = {}
        for node in graph.nodes:
            scores[node.id] = self.score_node(
                node,
                now=now,
                max_access=max_access,
                degree=degree,
                max_degree=max_degree,
                suppressed=suppressed,
            )

        # 2. Independent edge pruning (weak edges between low-scoring endpoints).
        surviving_edges = [
            e
            for e in graph.edges
            if not (
                scores.get(e.source, 1.0) < self.edge_score_threshold
                and scores.get(e.target, 1.0) < self.edge_score_threshold
            )
        ]

        # 3. Threshold pruning on nodes.
        kept_ids: Set[str] = {
            nid for nid, s in scores.items() if s >= self.threshold
        }

        # 4. Hard cap.
        if self.max_nodes is not None and len(kept_ids) > self.max_nodes:
            ranked = sorted(kept_ids, key=lambda nid: scores[nid], reverse=True)
            kept_ids = set(ranked[: self.max_nodes])

        # 5. Rebuild graph — drop forgotten nodes and dangling edges.
        surviving_nodes = [n for n in graph.nodes if n.id in kept_ids]
        surviving_edges = [
            e
            for e in surviving_edges
            if e.source in kept_ids and e.target in kept_ids
        ]

        # 6. Purge metadata for forgotten nodes.
        forgotten = set(scores.keys()) - kept_ids
        for nid in forgotten:
            self._meta.pop(nid, None)

        return MemoryGraph(nodes=surviving_nodes, edges=surviving_edges)

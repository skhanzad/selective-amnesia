"""Pydantic models for memory nodes, edges, and retrieved memory bundles."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    entity = "entity"
    fact = "fact"
    event = "event"
    user_preference = "user_preference"
    task = "task"
    summary = "summary"
    source = "source"
    belief = "belief"


class EdgeType(str, Enum):
    related_to = "related_to"
    refers_to = "refers_to"
    supports = "supports"
    contradicts = "contradicts"
    supersedes = "supersedes"
    caused_by = "caused_by"
    temporal_before = "temporal_before"
    derived_from = "derived_from"
    similar_to = "similar_to"


class MemoryNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    node_type: NodeType = NodeType.fact
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    turn_created: int = 0
    turn_last_accessed: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    def touch(self, turn: int) -> None:
        """Mark this node as accessed at the given turn."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        self.turn_last_accessed = turn

    def age_in_turns(self, current_turn: int) -> int:
        """How many turns since this node was last accessed."""
        return current_turn - self.turn_last_accessed


class MemoryEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.related_to
    weight: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedMemory(BaseModel):
    """A memory node bundled with its graph context for injection into the LLM."""
    node: MemoryNode
    edges: list[MemoryEdge] = Field(default_factory=list)
    neighbors: list[MemoryNode] = Field(default_factory=list)
    relevance_score: float = 0.0

    def to_context_string(self) -> str:
        """Format this memory for LLM context injection."""
        parts = [f"[{self.node.node_type.value}] {self.node.content}"]
        if self.edges:
            for edge in self.edges:
                # Find the neighbor this edge connects to
                other_id = (
                    edge.target_id if edge.source_id == self.node.id else edge.source_id
                )
                neighbor = next(
                    (n for n in self.neighbors if n.id == other_id), None
                )
                neighbor_label = neighbor.content[:60] if neighbor else other_id[:8]
                parts.append(f"  --{edge.edge_type.value}--> {neighbor_label}")
        return "\n".join(parts)

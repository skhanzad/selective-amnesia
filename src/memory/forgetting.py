"""Modular forgetting policies for selective amnesia.

Each policy scores nodes for removal (0.0 = keep, 1.0 = remove).
Policies are composable and configurable via YAML.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.memory.schemas import MemoryNode

logger = logging.getLogger(__name__)


class ForgettingPolicy(ABC):
    """Base class for forgetting policies."""

    @abstractmethod
    def score_for_removal(
        self, node: MemoryNode, current_turn: int
    ) -> float:
        """Return 0.0 (keep) to 1.0 (remove)."""
        ...


class NoForgetting(ForgettingPolicy):
    """Never forget anything."""

    def score_for_removal(self, node: MemoryNode, current_turn: int) -> float:
        return 0.0


class RecencyPolicy(ForgettingPolicy):
    """Forget nodes that haven't been accessed recently."""

    def __init__(self, decay_rate: float = 0.05, max_age: int = 100) -> None:
        self.decay_rate = decay_rate
        self.max_age = max_age

    def score_for_removal(self, node: MemoryNode, current_turn: int) -> float:
        age = node.age_in_turns(current_turn)
        return min(1.0, age * self.decay_rate)


class ImportancePolicy(ForgettingPolicy):
    """Forget low-importance nodes."""

    def score_for_removal(self, node: MemoryNode, current_turn: int) -> float:
        return 1.0 - node.importance


class HybridPolicy(ForgettingPolicy):
    """Weighted combination of recency, importance, and access frequency."""

    def __init__(
        self,
        recency_weight: float = 0.4,
        importance_weight: float = 0.4,
        access_weight: float = 0.2,
        decay_rate: float = 0.05,
    ) -> None:
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.access_weight = access_weight
        self.recency = RecencyPolicy(decay_rate=decay_rate)

    def score_for_removal(self, node: MemoryNode, current_turn: int) -> float:
        recency_score = self.recency.score_for_removal(node, current_turn)
        importance_score = 1.0 - node.importance
        access_score = 1.0 / (1.0 + node.access_count)
        return (
            self.recency_weight * recency_score
            + self.importance_weight * importance_score
            + self.access_weight * access_score
        )


# ── Policy registry ─────────────────────────────────────────────

POLICY_REGISTRY: dict[str, type[ForgettingPolicy]] = {
    "none": NoForgetting,
    "recency": RecencyPolicy,
    "importance": ImportancePolicy,
    "hybrid": HybridPolicy,
}


def get_policy(name: str, config: dict[str, Any]) -> ForgettingPolicy:
    """Instantiate a forgetting policy from config."""
    cls = POLICY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown forgetting policy: {name!r}. Available: {list(POLICY_REGISTRY)}")

    forget_cfg = config.get("forgetting", {})
    if name == "recency":
        return RecencyPolicy(
            decay_rate=forget_cfg.get("decay_rate", config.get("memory", {}).get("decay_rate", 0.05)),
        )
    elif name == "hybrid":
        return HybridPolicy(
            recency_weight=forget_cfg.get("recency_weight", 0.4),
            importance_weight=forget_cfg.get("importance_weight", 0.4),
            access_weight=forget_cfg.get("access_frequency_weight", 0.2),
            decay_rate=config.get("memory", {}).get("decay_rate", 0.05),
        )
    else:
        return cls()


# ── Apply forgetting ─────────────────────────────────────────────

def apply_forgetting(
    nodes: list[MemoryNode],
    policy: ForgettingPolicy,
    current_turn: int,
    budget_target: int,
    min_importance: float = 0.3,
) -> list[str]:
    """Score all nodes and return IDs of those to remove.

    Only removes nodes if over budget. Never removes nodes with
    importance >= min_importance.
    """
    if len(nodes) <= budget_target:
        return []

    scored: list[tuple[str, float]] = []
    for node in nodes:
        if node.importance >= min_importance:
            continue
        score = policy.score_for_removal(node, current_turn)
        scored.append((node.id, score))

    # Sort by score descending (highest = most removable)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Remove enough to get to budget
    n_to_remove = len(nodes) - budget_target
    to_remove = [nid for nid, _ in scored[:n_to_remove]]

    logger.info(
        "Forgetting %d nodes (budget_target=%d, current=%d)",
        len(to_remove),
        budget_target,
        len(nodes),
    )
    return to_remove

"""Tests for memory schemas, graph store, forgetting policies, and retrievers."""

import pytest

from src.memory.schemas import EdgeType, MemoryEdge, MemoryNode, NodeType, RetrievedMemory
from src.memory.graph_store import GraphStore
from src.memory.forgetting import (
    HybridPolicy,
    ImportancePolicy,
    RecencyPolicy,
    apply_forgetting,
)
from src.memory.retriever import FlatRetriever, GraphRetriever


# ── Schema tests ─────────────────────────────────────────────────

def test_memory_node_defaults():
    node = MemoryNode(content="test fact")
    assert node.content == "test fact"
    assert node.node_type == NodeType.fact
    assert node.importance == 0.5
    assert node.access_count == 0
    assert node.enabled is True
    assert node.id  # UUID generated


def test_memory_node_touch():
    node = MemoryNode(content="test", turn_created=0, turn_last_accessed=0)
    node.touch(turn=5)
    assert node.access_count == 1
    assert node.turn_last_accessed == 5
    node.touch(turn=8)
    assert node.access_count == 2
    assert node.turn_last_accessed == 8


def test_memory_node_age():
    node = MemoryNode(content="test", turn_last_accessed=3)
    assert node.age_in_turns(10) == 7


def test_retrieved_memory_context_string():
    node = MemoryNode(content="Alice likes coffee", node_type=NodeType.user_preference)
    rm = RetrievedMemory(node=node)
    ctx = rm.to_context_string()
    assert "user_preference" in ctx
    assert "Alice likes coffee" in ctx


# ── GraphStore tests ─────────────────────────────────────────────

def test_graph_store_add_get():
    store = GraphStore()
    node = MemoryNode(content="earth is round")
    nid = store.add_node(node)
    retrieved = store.get_node(nid)
    assert retrieved is not None
    assert retrieved.content == "earth is round"


def test_graph_store_remove():
    store = GraphStore()
    node = MemoryNode(content="temp")
    nid = store.add_node(node)
    assert store.remove_node(nid) is True
    assert store.get_node(nid) is None
    assert store.node_count() == 0


def test_graph_store_disable():
    store = GraphStore()
    node = MemoryNode(content="quarantine me")
    nid = store.add_node(node)
    store.disable_node(nid)
    assert store.node_count(include_disabled=False) == 0
    assert store.node_count(include_disabled=True) == 1


def test_graph_store_edges():
    store = GraphStore()
    a = MemoryNode(content="A")
    b = MemoryNode(content="B")
    store.add_node(a)
    store.add_node(b)

    edge = MemoryEdge(source_id=a.id, target_id=b.id, edge_type=EdgeType.supports)
    store.add_edge(edge)

    edges = store.get_edges(a.id)
    assert len(edges) == 1
    assert edges[0].edge_type == EdgeType.supports


def test_graph_store_neighbors():
    store = GraphStore()
    a = MemoryNode(content="A")
    b = MemoryNode(content="B")
    c = MemoryNode(content="C")
    store.add_node(a)
    store.add_node(b)
    store.add_node(c)

    store.add_edge(MemoryEdge(source_id=a.id, target_id=b.id))
    store.add_edge(MemoryEdge(source_id=b.id, target_id=c.id))

    depth1 = store.get_neighbors(a.id, depth=1)
    assert len(depth1) == 1
    assert depth1[0].id == b.id

    depth2 = store.get_neighbors(a.id, depth=2)
    assert len(depth2) == 2
    ids = {n.id for n in depth2}
    assert b.id in ids
    assert c.id in ids


def test_graph_store_search_by_content():
    store = GraphStore()
    store.add_node(MemoryNode(content="Alice likes coffee"))
    store.add_node(MemoryNode(content="Bob likes tea"))
    store.add_node(MemoryNode(content="Charlie likes coffee too"))

    results = store.search_by_content("coffee")
    assert len(results) == 2


def test_graph_store_serialization():
    store = GraphStore()
    a = MemoryNode(content="A", importance=0.9)
    b = MemoryNode(content="B", importance=0.3)
    store.add_node(a)
    store.add_node(b)
    store.add_edge(MemoryEdge(source_id=a.id, target_id=b.id, edge_type=EdgeType.contradicts))

    data = store.to_dict()
    restored = GraphStore.from_dict(data)

    assert restored.node_count() == 2
    assert restored.get_node(a.id).content == "A"
    assert restored.edge_count() == 1


# ── Forgetting policy tests ─────────────────────────────────────

def test_recency_policy():
    policy = RecencyPolicy(decay_rate=0.1)
    old = MemoryNode(content="old", turn_last_accessed=0, importance=0.2)
    new = MemoryNode(content="new", turn_last_accessed=9, importance=0.2)

    score_old = policy.score_for_removal(old, current_turn=10)
    score_new = policy.score_for_removal(new, current_turn=10)
    assert score_old > score_new


def test_importance_policy():
    policy = ImportancePolicy()
    low = MemoryNode(content="low", importance=0.1)
    high = MemoryNode(content="high", importance=0.9)

    assert policy.score_for_removal(low, 0) > policy.score_for_removal(high, 0)


def test_hybrid_policy():
    policy = HybridPolicy()
    old_low = MemoryNode(content="old low", turn_last_accessed=0, importance=0.1, access_count=0)
    new_high = MemoryNode(content="new high", turn_last_accessed=9, importance=0.9, access_count=5)

    score_old = policy.score_for_removal(old_low, current_turn=10)
    score_new = policy.score_for_removal(new_high, current_turn=10)
    assert score_old > score_new


def test_apply_forgetting_under_budget():
    nodes = [MemoryNode(content=f"node {i}", importance=0.2) for i in range(5)]
    removed = apply_forgetting(nodes, RecencyPolicy(), current_turn=10, budget_target=10)
    assert removed == []  # under budget, nothing removed


def test_apply_forgetting_over_budget():
    nodes = [
        MemoryNode(content=f"node {i}", importance=0.2, turn_last_accessed=i)
        for i in range(20)
    ]
    removed = apply_forgetting(
        nodes, RecencyPolicy(decay_rate=0.1), current_turn=20, budget_target=10
    )
    assert len(removed) == 10  # should remove 10 to reach budget


def test_apply_forgetting_preserves_important():
    nodes = [
        MemoryNode(content=f"important {i}", importance=0.9, turn_last_accessed=0)
        for i in range(15)
    ]
    removed = apply_forgetting(
        nodes, RecencyPolicy(), current_turn=100, budget_target=5, min_importance=0.5
    )
    assert removed == []  # all nodes are important, none removed


# ── Retriever tests ──────────────────────────────────────────────

def test_flat_retriever():
    store = GraphStore()
    for i in range(5):
        n = MemoryNode(content=f"memory {i}", turn_last_accessed=i)
        store.add_node(n)

    retriever = FlatRetriever()
    results = retriever.retrieve("anything", store, max_results=3, current_turn=5)
    assert len(results) == 3


def test_graph_retriever_keyword_match():
    store = GraphStore()
    store.add_node(MemoryNode(content="Alice loves Python programming"))
    store.add_node(MemoryNode(content="Bob enjoys hiking mountains"))
    store.add_node(MemoryNode(content="Python is a great language"))

    retriever = GraphRetriever()
    results = retriever.retrieve("Python programming", store, max_results=5, current_turn=1)
    assert len(results) >= 1
    # Best match should mention Python
    assert "Python" in results[0].node.content


def test_graph_retriever_empty_store():
    store = GraphStore()
    retriever = GraphRetriever()
    results = retriever.retrieve("anything", store, max_results=5, current_turn=1)
    assert results == []

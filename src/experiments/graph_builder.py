"""Build MemoryGraph from benchmark conversations.

Processes dialogue sessions in batches, extracts nodes/edges via the LLM,
and optionally applies the ForgetPolicy after each session.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from experiments.config import CACHE_DIR, DEFAULT_MODEL, DEFAULT_PROVIDER, EXTRACTION_BATCH_TURNS
from experiments.data_loaders import (
    LoCoMoSample,
    LongMemInstance,
    locomo_session_to_text,
    longmem_session_to_text,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.graph_store import MemoryGraph
from memory.schemas import MemoryNode, MemoryEdge, NodeType, EdgeType
from memory.forgetting import ForgetPolicy
from agent.llm import build_llm
from agent.nodes import (
    ExtractedGraph,
    _EXTRACTION_SYSTEM_PROMPT,
    _convert_extracted,
    _forget_policy,
)


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
# Core extraction — feed text to the LLM and merge into a graph
# =====================================================================

def extract_and_merge(
    text: str,
    existing: MemoryGraph,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
) -> MemoryGraph:
    """Run structured extraction on *text* and merge into *existing*."""
    llm = build_llm(model=model, provider=provider)
    structured_llm = llm.with_structured_output(ExtractedGraph)

    messages = [
        SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Extract a memory graph from this conversation:\n\n{text}"),
    ]

    try:
        extracted: ExtractedGraph = structured_llm.invoke(messages)
    except Exception as e:
        print(f"  [extraction error: {e}]")
        return existing

    existing_node_ids = {n.id for n in existing.nodes}
    existing_edge_ids = {e.id for e in existing.edges}

    new = _convert_extracted(extracted, existing_node_ids)

    nodes = list(existing.nodes) + new.nodes
    edges = list(existing.edges) + [e for e in new.edges if e.id not in existing_edge_ids]
    return MemoryGraph(nodes=nodes, edges=edges)


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

    for sess in sample.sessions:
        text = locomo_session_to_text(sess, sample.speaker_a, sample.speaker_b)
        graph = extract_and_merge(text, graph, model=model, provider=provider)
        policy.track_many(graph.nodes)

        if forget_preset != "none":
            graph = policy.apply(graph)

        print(f"  session {sess.session_num}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

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

    for sess in instance.sessions:
        text = longmem_session_to_text(sess)
        graph = extract_and_merge(text, graph, model=model, provider=provider)
        policy.track_many(graph.nodes)

        if forget_preset != "none":
            graph = policy.apply(graph)

    _save_graph(cache_path, graph)
    return graph

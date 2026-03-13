"""Build and compile the LangGraph agent pipeline."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    apply_forgetting_node,
    extract_memories,
    generate_response,
    retrieve_memories,
)
from src.agent.state import GraphState
from src.memory.forgetting import get_policy
from src.memory.graph_store import GraphStore
from src.memory.retriever import get_retriever


def _build_llm(config: dict[str, Any]) -> Any:
    """Create an LLM instance from config."""
    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("provider", "ollama")
    model = llm_cfg.get("model", "llama3.2")
    temperature = llm_cfg.get("temperature", 0.7)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'openai' or 'ollama'.")


def build_agent_graph(
    config: dict[str, Any],
    graph_store: GraphStore | None = None,
) -> tuple[Any, GraphStore]:
    """Build the LangGraph pipeline and return (compiled_graph, graph_store).

    Pipeline: retrieve -> generate -> extract -> forget
    """
    if graph_store is None:
        graph_store = GraphStore()

    llm = _build_llm(config)

    retriever = get_retriever(
        config.get("retrieval", {}).get("mode", "graph"),
        config,
    )
    policy = get_policy(
        config.get("forgetting", {}).get("policy", "hybrid"),
        config,
    )

    # Build the state graph
    workflow = StateGraph(GraphState)

    # Add nodes with dependencies bound via partial
    workflow.add_node(
        "retrieve",
        partial(retrieve_memories, retriever=retriever, graph_store=graph_store, config=config),
    )
    workflow.add_node(
        "generate",
        partial(generate_response, llm=llm, config=config),
    )
    workflow.add_node(
        "extract",
        partial(extract_memories, llm=llm, graph_store=graph_store, config=config),
    )
    workflow.add_node(
        "forget",
        partial(apply_forgetting_node, graph_store=graph_store, policy=policy, config=config),
    )

    # Linear pipeline: retrieve -> generate -> extract -> forget
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "extract")
    workflow.add_edge("extract", "forget")
    workflow.add_edge("forget", END)

    compiled = workflow.compile()
    return compiled, graph_store

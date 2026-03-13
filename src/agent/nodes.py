"""LangGraph node functions for the retrieve-generate-extract-forget pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.memory.forgetting import ForgettingPolicy, apply_forgetting
from src.memory.graph_store import GraphStore
from src.memory.retriever import MemoryRetriever
from src.memory.schemas import MemoryNode, NodeType

logger = logging.getLogger(__name__)


def retrieve_memories(
    state: dict[str, Any],
    *,
    retriever: MemoryRetriever,
    graph_store: GraphStore,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Retrieve relevant memories for the current user message."""
    messages = state.get("messages", [])
    current_turn = state.get("current_turn", 0)

    # Get the last user message
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"retrieved_memories": []}

    ret_cfg = config.get("retrieval", {})
    max_results = ret_cfg.get("max_results", 10)

    results = retriever.retrieve(
        query=last_user_msg,
        graph_store=graph_store,
        max_results=max_results,
        current_turn=current_turn,
    )

    # Touch each retrieved node
    for rm in results:
        rm.node.touch(current_turn)

    logger.info("Retrieved %d memories for query: %s", len(results), last_user_msg[:50])

    # Serialize for state (LangGraph state must be JSON-serializable)
    return {
        "retrieved_memories": [rm.model_dump(mode="json") for rm in results],
    }


def generate_response(
    state: dict[str, Any],
    *,
    llm: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate an LLM response using conversation history and retrieved memories."""
    agent_cfg = config.get("agent", {})
    system_prompt = agent_cfg.get("system_prompt", "You are a helpful assistant.")

    # Build memory context
    retrieved = state.get("retrieved_memories", [])
    memory_context = ""
    if retrieved:
        memory_lines = []
        for rm in retrieved:
            node = rm.get("node", {})
            node_type = node.get("node_type", "fact")
            content = node.get("content", "")
            memory_lines.append(f"- [{node_type}] {content}")

            # Include edge context
            for edge in rm.get("edges", []):
                edge_type = edge.get("edge_type", "related_to")
                # Find neighbor content
                target_id = edge.get("target_id", "")
                for neighbor in rm.get("neighbors", []):
                    if neighbor.get("id") == target_id:
                        memory_lines.append(
                            f"    --{edge_type}--> {neighbor.get('content', '')[:60]}"
                        )
                        break

        memory_context = "\n\nRetrieved Memories:\n" + "\n".join(memory_lines)

    full_system = system_prompt.strip() + memory_context

    # Build message list
    msgs = [SystemMessage(content=full_system)]
    for msg in state.get("messages", []):
        msgs.append(msg)

    response = llm.invoke(msgs)

    return {"messages": [response]}


def extract_memories(
    state: dict[str, Any],
    *,
    llm: Any,
    graph_store: GraphStore,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Extract new memories from the latest conversation turn."""
    messages = state.get("messages", [])
    current_turn = state.get("current_turn", 0)

    if len(messages) < 2:
        return {"new_memories": []}

    # Get last user + assistant messages
    last_user = ""
    last_assistant = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not last_assistant:
            last_assistant = msg.content
        elif isinstance(msg, HumanMessage) and not last_user:
            last_user = msg.content
        if last_user and last_assistant:
            break

    if not last_user:
        return {"new_memories": []}

    agent_cfg = config.get("agent", {})
    extraction_prompt = agent_cfg.get("extraction_prompt", "")

    # Build existing memory summary so the LLM can link to them
    existing_nodes = graph_store.get_all_nodes()
    existing_summary = ""
    if existing_nodes:
        lines = ["Existing memories (you may reference these by index for edges):"]
        for i, n in enumerate(existing_nodes[:20]):  # cap to avoid prompt bloat
            lines.append(f"  [{i}] ({n.node_type.value}) {n.content[:80]}")
        existing_summary = "\\n".join(lines)

    prompt = extraction_prompt.format(
        user_message=last_user,
        assistant_message=last_assistant,
        existing_memories=existing_summary if existing_summary else "No existing memories yet.",
    )

    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        raw = result.content.strip()
        logger.debug("Memory extraction raw LLM output: %s", raw[:500])

        # Try to parse JSON from the response
        # Handle cases where LLM wraps JSON in markdown code blocks
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        # Try to find a JSON array in the response
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]

        memories_data = json.loads(raw)
        if not isinstance(memories_data, list):
            memories_data = [memories_data]

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse memory extraction JSON: %s\nRaw: %s", e, raw[:300])
        return {"new_memories": []}
    except Exception as e:
        logger.warning("Memory extraction failed: %s", e, exc_info=True)
        return {"new_memories": []}

    # Create MemoryNode objects and add to graph
    new_nodes: list[dict[str, Any]] = []
    default_importance = config.get("memory", {}).get("default_importance", 0.5)

    for mem in memories_data:
        if not isinstance(mem, dict) or "content" not in mem:
            continue

        node_type_str = mem.get("node_type", "fact")
        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            node_type = NodeType.fact

        importance = mem.get("importance", default_importance)
        if not isinstance(importance, (int, float)):
            importance = default_importance

        node = MemoryNode(
            content=mem["content"],
            node_type=node_type,
            importance=min(1.0, max(0.0, float(importance))),
            turn_created=current_turn,
            turn_last_accessed=current_turn,
        )
        graph_store.add_node(node)
        new_nodes.append(node.model_dump(mode="json"))

    logger.info("Extracted %d new memories at turn %d", len(new_nodes), current_turn)
    return {"new_memories": new_nodes}


def apply_forgetting_node(
    state: dict[str, Any],
    *,
    graph_store: GraphStore,
    policy: ForgettingPolicy,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply forgetting policy to prune the memory graph."""
    current_turn = state.get("current_turn", 0)
    forget_cfg = config.get("forgetting", {})
    budget_target = forget_cfg.get("budget_target", 150)
    min_importance = forget_cfg.get("min_importance_to_keep", 0.3)
    run_every = forget_cfg.get("run_every_n_turns", 1)

    if current_turn % run_every != 0:
        return {"forgotten_ids": []}

    nodes = graph_store.get_all_nodes()
    to_remove = apply_forgetting(
        nodes=nodes,
        policy=policy,
        current_turn=current_turn,
        budget_target=budget_target,
        min_importance=min_importance,
    )

    for nid in to_remove:
        graph_store.remove_node(nid)

    if to_remove:
        logger.info("Forgot %d memories at turn %d", len(to_remove), current_turn)

    return {"forgotten_ids": to_remove}

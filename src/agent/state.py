"""LangGraph state schema for the memory agent."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(dict):
    """State flowing through the LangGraph pipeline.

    Fields:
        messages: Conversation history (accumulated via add_messages reducer).
        retrieved_memories: Memories retrieved for the current turn.
        new_memories: Memories extracted from the current turn.
        forgotten_ids: Node IDs removed by forgetting this turn.
        current_turn: Integer turn counter.
    """
    pass


# Use TypedDict-style annotations for LangGraph
from typing import TypedDict


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved_memories: list[dict[str, Any]]
    new_memories: list[dict[str, Any]]
    forgotten_ids: list[str]
    current_turn: int

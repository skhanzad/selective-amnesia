from pathlib import Path
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from agent.state import State
from agent.graph import graph
from memory import MemoryGraph

MEMORY_PATH = Path(__file__).parent / "memory_store.json"


def load_memory() -> MemoryGraph:
    """Load the persisted memory graph from disk, or return an empty one."""
    if not MEMORY_PATH.exists():
        return MemoryGraph(nodes=[], edges=[])
    text = MEMORY_PATH.read_text().strip()
    if not text:
        return MemoryGraph(nodes=[], edges=[])
    try:
        return MemoryGraph.model_validate_json(text)
    except (ValidationError, ValueError):
        return MemoryGraph(nodes=[], edges=[])


def save_memory(memory: MemoryGraph) -> None:
    """Persist the memory graph to disk."""
    MEMORY_PATH.write_text(memory.model_dump_json(indent=2))


def main():
    g = graph().compile()
    memory = load_memory()
    history: List[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant with access to a persistent memory graph."),
    ]

    print(f"Loaded memory: {len(memory.nodes)} nodes, {len(memory.edges)} edges")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nSaving memory and exiting.")
            save_memory(memory)
            break

        user_msg = HumanMessage(content=user_input)
        history.append(user_msg)

        state = State(
            messages=list(history),
            memory=memory,
        )
        state = g.invoke(state)

        # Find the AI response (last ai message in the returned state).
        ai_msg = None
        for msg in reversed(state["messages"]):
            if msg.type == "ai":
                ai_msg = msg
                break

        if ai_msg:
            print(f"AI: {ai_msg.content}")
            history.append(ai_msg)

        # Carry the (possibly pruned) memory forward and persist it.
        memory = state["memory"]
        save_memory(memory)
        print(f"  [{len(memory.nodes)} nodes, {len(memory.edges)} edges in memory]")


if __name__ == "__main__":
    main()
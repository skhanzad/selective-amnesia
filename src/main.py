"""CLI entrypoint: interactive conversation loop with memory agent."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.agent.graph import build_agent_graph
from src.memory.graph_store import GraphStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAVE_PATH = "data/memory_graph.json"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def print_memory_stats(graph_store: GraphStore) -> None:
    nodes = graph_store.get_all_nodes(include_disabled=True)
    enabled = [n for n in nodes if n.enabled]
    disabled = [n for n in nodes if not n.enabled]
    print(f"\n--- Memory Stats ---")
    print(f"  Total nodes: {len(nodes)} (enabled: {len(enabled)}, disabled: {len(disabled)})")
    print(f"  Edges: {graph_store.edge_count()}")
    # Type distribution
    from collections import Counter
    type_counts = Counter(n.node_type.value for n in enabled)
    if type_counts:
        print(f"  Types: {dict(type_counts)}")
    print()


def print_all_memories(graph_store: GraphStore) -> None:
    nodes = graph_store.get_all_nodes()
    if not nodes:
        print("\n  (no memories stored)\n")
        return
    print(f"\n--- All Memories ({len(nodes)}) ---")
    for n in sorted(nodes, key=lambda x: x.turn_created):
        status = "" if n.enabled else " [DISABLED]"
        print(f"  [{n.node_type.value}] {n.content} (imp={n.importance:.2f}, turn={n.turn_created}){status}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective Amnesia - Memory Agent")
    parser.add_argument(
        "--config", default="configs/base.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--load", default=None, help="Path to load saved memory graph"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("src").setLevel(logging.DEBUG)

    load_dotenv()
    config = load_config(args.config)

    # Load or create graph store
    graph_store = None
    load_path = args.load or SAVE_PATH
    if Path(load_path).exists():
        try:
            graph_store = GraphStore.load(load_path)
            logger.info("Loaded memory graph from %s (%d nodes)", load_path, graph_store.node_count())
        except Exception as e:
            logger.warning("Failed to load graph from %s: %s", load_path, e)

    compiled_graph, graph_store = build_agent_graph(config, graph_store)

    print("Selective Amnesia - Graph Memory Agent")
    print("Commands: /memories, /stats, /save, /clear, quit")
    print("-" * 45)

    current_turn = 0
    all_messages: list = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            break

        # Special commands
        if user_input == "/memories":
            print_all_memories(graph_store)
            continue
        if user_input == "/stats":
            print_memory_stats(graph_store)
            continue
        if user_input == "/save":
            graph_store.save(SAVE_PATH)
            print(f"  Saved to {SAVE_PATH}")
            continue
        if user_input == "/clear":
            graph_store = GraphStore()
            compiled_graph, graph_store = build_agent_graph(config, graph_store)
            all_messages = []
            current_turn = 0
            print("  Memory cleared.")
            continue

        current_turn += 1
        all_messages.append(HumanMessage(content=user_input))

        state = {
            "messages": all_messages,
            "retrieved_memories": [],
            "new_memories": [],
            "forgotten_ids": [],
            "current_turn": current_turn,
        }

        try:
            result = compiled_graph.invoke(state)
        except Exception as e:
            logger.error("Agent error: %s", e)
            print(f"\n  Error: {e}")
            continue

        # Extract AI response
        response_msgs = result.get("messages", [])
        if response_msgs:
            ai_msg = response_msgs[-1]
            content = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
            print(f"\nAssistant: {content}")
            all_messages = list(result["messages"])
        else:
            print("\n  (no response)")

        # Print turn stats
        n_retrieved = len(result.get("retrieved_memories", []))
        n_new = len(result.get("new_memories", []))
        n_forgot = len(result.get("forgotten_ids", []))
        total = graph_store.node_count()
        print(
            f"  [mem: {total} | retrieved: {n_retrieved} | +{n_new} | -{n_forgot}]"
        )

    # Save on exit
    graph_store.save(SAVE_PATH)
    print(f"\nMemory saved to {SAVE_PATH}. Goodbye.")


if __name__ == "__main__":
    main()

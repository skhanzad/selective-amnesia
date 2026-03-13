"""Streamlit UI for visualizing and editing the memory graph."""

from __future__ import annotations

import json
import streamlit as st
from pathlib import Path
from pyvis.network import Network

from src.memory.graph_store import GraphStore
from src.memory.schemas import EdgeType, MemoryEdge, MemoryNode, NodeType

GRAPH_PATH = "data/memory_graph.json"

# Color scheme per node type
NODE_COLORS = {
    "entity": "#4A90D9",
    "fact": "#50C878",
    "event": "#FFB347",
    "user_preference": "#FF6B6B",
    "task": "#9B59B6",
    "summary": "#1ABC9C",
    "source": "#95A5A6",
    "belief": "#F39C12",
}

EDGE_COLORS = {
    "contradicts": "#FF0000",
    "supersedes": "#FF6600",
    "supports": "#00AA00",
    "related_to": "#888888",
    "refers_to": "#6699CC",
    "caused_by": "#CC6699",
    "temporal_before": "#9966CC",
    "derived_from": "#669966",
    "similar_to": "#CCCC00",
}


def load_store() -> GraphStore:
    path = Path(GRAPH_PATH)
    if path.exists():
        return GraphStore.load(str(path))
    return GraphStore()


def save_store(store: GraphStore) -> None:
    store.save(GRAPH_PATH)


def render_graph(store: GraphStore, filter_node_types: list[str], filter_edge_types: list[str], show_disabled: bool) -> str:
    """Render the graph as an interactive HTML string using pyvis."""
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        select_menu=False,
        filter_menu=False,
    )

    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.01,
                "springLength": 150,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100}
        },
        "nodes": {
            "font": {"size": 14, "face": "monospace"},
            "borderWidth": 2,
            "shadow": true
        },
        "edges": {
            "font": {"size": 10, "face": "monospace", "color": "#aaaaaa"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
            "shadow": true
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    nodes = store.get_all_nodes(include_disabled=show_disabled)

    # Filter by node type
    if filter_node_types:
        nodes = [n for n in nodes if n.node_type.value in filter_node_types]

    node_ids = {n.id for n in nodes}

    for node in nodes:
        color = NODE_COLORS.get(node.node_type.value, "#888888")
        opacity = "40" if not node.enabled else ""
        label = node.content[:40] + ("..." if len(node.content) > 40 else "")
        title = (
            f"<b>{node.node_type.value}</b><br>"
            f"{node.content}<br><br>"
            f"importance: {node.importance:.2f}<br>"
            f"access_count: {node.access_count}<br>"
            f"turn_created: {node.turn_created}<br>"
            f"enabled: {node.enabled}<br>"
            f"id: {node.id[:8]}..."
        )
        size = 15 + node.importance * 25
        border_color = "#ff4444" if not node.enabled else color

        net.add_node(
            node.id,
            label=label,
            title=title,
            color={"background": color + opacity, "border": border_color},
            size=size,
            shape="dot",
        )

    # Add edges
    for node in nodes:
        edges = store.get_edges(node.id, direction="out")
        for edge in edges:
            if edge.target_id not in node_ids:
                continue
            if filter_edge_types and edge.edge_type.value not in filter_edge_types:
                continue

            edge_color = EDGE_COLORS.get(edge.edge_type.value, "#888888")
            width = 1.5 if edge.edge_type.value in ("contradicts", "supersedes") else 1.0

            net.add_edge(
                edge.source_id,
                edge.target_id,
                label=edge.edge_type.value,
                color=edge_color,
                width=width,
                title=f"{edge.edge_type.value} (weight: {edge.weight})",
            )

    html = net.generate_html()
    return html


def main() -> None:
    st.set_page_config(page_title="Selective Amnesia - Memory Graph", layout="wide")
    st.title("Memory Graph Explorer")

    # Initialize session state
    if "store" not in st.session_state:
        st.session_state.store = load_store()

    store: GraphStore = st.session_state.store

    # ── Sidebar: filters and controls ────────────────────────────
    with st.sidebar:
        st.header("Controls")

        if st.button("Reload from disk", use_container_width=True):
            st.session_state.store = load_store()
            st.rerun()

        if st.button("Save to disk", use_container_width=True):
            save_store(store)
            st.success("Saved!")

        st.divider()
        st.header("Filters")

        all_node_types = [t.value for t in NodeType]
        filter_node_types = st.multiselect(
            "Node types", all_node_types, default=all_node_types
        )

        all_edge_types = [t.value for t in EdgeType]
        filter_edge_types = st.multiselect(
            "Edge types", all_edge_types, default=all_edge_types
        )

        show_disabled = st.checkbox("Show disabled nodes", value=False)

        st.divider()
        st.header("Stats")
        total = store.node_count(include_disabled=True)
        enabled = store.node_count(include_disabled=False)
        st.metric("Total nodes", total)
        st.metric("Enabled", enabled)
        st.metric("Disabled", total - enabled)
        st.metric("Edges", store.edge_count())

    # ── Main area: tabs ──────────────────────────────────────────
    tab_graph, tab_nodes, tab_edit, tab_add = st.tabs(
        ["Graph", "Node Table", "Edit / Delete", "Add Node / Edge"]
    )

    # ── Tab 1: Interactive Graph ─────────────────────────────────
    with tab_graph:
        if store.node_count(include_disabled=True) == 0:
            st.info("No memories yet. Run the agent to populate the graph.")
        else:
            html = render_graph(store, filter_node_types, filter_edge_types, show_disabled)
            st.components.v1.html(html, height=620, scrolling=False)

            # Legend
            cols = st.columns(len(NODE_COLORS))
            for col, (ntype, color) in zip(cols, NODE_COLORS.items()):
                col.markdown(
                    f'<span style="color:{color}">&#9679;</span> {ntype}',
                    unsafe_allow_html=True,
                )

    # ── Tab 2: Node table ────────────────────────────────────────
    with tab_nodes:
        nodes = store.get_all_nodes(include_disabled=show_disabled)
        if not nodes:
            st.info("No nodes to display.")
        else:
            rows = []
            for n in sorted(nodes, key=lambda x: x.turn_created, reverse=True):
                rows.append({
                    "id": n.id[:8],
                    "type": n.node_type.value,
                    "content": n.content,
                    "importance": n.importance,
                    "access_count": n.access_count,
                    "turn": n.turn_created,
                    "enabled": n.enabled,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Tab 3: Edit / Delete ─────────────────────────────────────
    with tab_edit:
        nodes = store.get_all_nodes(include_disabled=True)
        if not nodes:
            st.info("No nodes to edit.")
        else:
            node_options = {
                f"[{n.node_type.value}] {n.content[:50]} ({n.id[:8]})": n.id
                for n in nodes
            }
            selected_label = st.selectbox("Select a node", list(node_options.keys()))

            if selected_label:
                node_id = node_options[selected_label]
                node = store.get_node(node_id)

                if node:
                    st.subheader("Edit Node")

                    new_content = st.text_area("Content", value=node.content, key="edit_content")
                    col1, col2 = st.columns(2)
                    with col1:
                        new_type = st.selectbox(
                            "Node type",
                            [t.value for t in NodeType],
                            index=[t.value for t in NodeType].index(node.node_type.value),
                            key="edit_type",
                        )
                    with col2:
                        new_importance = st.slider(
                            "Importance", 0.0, 1.0, node.importance, 0.05, key="edit_imp"
                        )

                    new_enabled = st.checkbox("Enabled", value=node.enabled, key="edit_enabled")

                    # Show edges
                    edges = store.get_edges(node_id)
                    if edges:
                        st.subheader("Edges")
                        for edge in edges:
                            direction = "outgoing" if edge.source_id == node_id else "incoming"
                            other_id = edge.target_id if edge.source_id == node_id else edge.source_id
                            other = store.get_node(other_id)
                            other_label = other.content[:40] if other else other_id[:8]
                            st.text(f"  {direction}: --{edge.edge_type.value}--> {other_label}")

                    col_save, col_toggle, col_delete = st.columns(3)

                    with col_save:
                        if st.button("Save changes", type="primary", use_container_width=True):
                            store.update_node(
                                node_id,
                                content=new_content,
                                node_type=NodeType(new_type),
                                importance=new_importance,
                                enabled=new_enabled,
                            )
                            save_store(store)
                            st.success("Node updated and saved.")
                            st.rerun()

                    with col_toggle:
                        label = "Disable" if node.enabled else "Enable"
                        if st.button(label, use_container_width=True):
                            if node.enabled:
                                store.disable_node(node_id)
                            else:
                                store.enable_node(node_id)
                            save_store(store)
                            st.success(f"Node {label.lower()}d.")
                            st.rerun()

                    with col_delete:
                        if st.button("Delete permanently", type="secondary", use_container_width=True):
                            store.remove_node(node_id)
                            save_store(store)
                            st.warning("Node deleted.")
                            st.rerun()

    # ── Tab 4: Add node / edge ───────────────────────────────────
    with tab_add:
        st.subheader("Add New Node")
        add_content = st.text_area("Content", key="add_content")
        col1, col2 = st.columns(2)
        with col1:
            add_type = st.selectbox("Node type", [t.value for t in NodeType], key="add_type")
        with col2:
            add_importance = st.slider("Importance", 0.0, 1.0, 0.5, 0.05, key="add_imp")

        if st.button("Add node", type="primary"):
            if add_content.strip():
                node = MemoryNode(
                    content=add_content.strip(),
                    node_type=NodeType(add_type),
                    importance=add_importance,
                )
                store.add_node(node)
                save_store(store)
                st.success(f"Added node: {node.id[:8]}")
                st.rerun()
            else:
                st.error("Content cannot be empty.")

        st.divider()
        st.subheader("Add New Edge")

        nodes = store.get_all_nodes(include_disabled=True)
        if len(nodes) < 2:
            st.info("Need at least 2 nodes to create an edge.")
        else:
            node_labels = {
                f"[{n.node_type.value}] {n.content[:50]} ({n.id[:8]})": n.id
                for n in nodes
            }
            col1, col2 = st.columns(2)
            with col1:
                source_label = st.selectbox("Source", list(node_labels.keys()), key="edge_src")
            with col2:
                target_label = st.selectbox("Target", list(node_labels.keys()), key="edge_tgt")

            edge_type = st.selectbox("Edge type", [t.value for t in EdgeType], key="edge_type")

            if st.button("Add edge", type="primary"):
                src_id = node_labels[source_label]
                tgt_id = node_labels[target_label]
                if src_id == tgt_id:
                    st.error("Source and target must be different.")
                else:
                    edge = MemoryEdge(
                        source_id=src_id,
                        target_id=tgt_id,
                        edge_type=EdgeType(edge_type),
                    )
                    try:
                        store.add_edge(edge)
                        save_store(store)
                        st.success("Edge added!")
                        st.rerun()
                    except KeyError as e:
                        st.error(str(e))


if __name__ == "__main__":
    main()

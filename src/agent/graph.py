from langgraph.graph import StateGraph, START, END
from agent.state import State
from nodes import retrieve_node, generate_node, build_graph_store, forget_node


def graph() -> StateGraph:
    g = StateGraph(State)
    g.add_node("retrieve_node", retrieve_node)
    g.add_node("generate_node", generate_node)
    g.add_node("build_graph_store", build_graph_store)
    g.add_node("forget_node", forget_node)
    g.add_edge(START, "retrieve_node")
    g.add_edge("retrieve_node", "generate_node")
    g.add_edge("generate_node", "build_graph_store")
    g.add_edge("build_graph_store", "forget_node")
    g.add_edge("forget_node", END)
    return g
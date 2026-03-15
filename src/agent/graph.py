from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from state import State
from nodes import retrieve_node, generate_node, extract_node, forget_node




def graph() -> StateGraph:
    return StateGraph(
        nodes=[
            START,
            END,
            retrieve_node,
            generate_node,
            extract_node,
            forget_node,
        ],
        edges=[
            (START, retrieve_node),
            (retrieve_node, generate_node),
            (generate_node, extract_node),
            (extract_node, forget_node),
            (forget_node, END),
        ],
    )
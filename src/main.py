from agent.state import State
from agent.state import trimmed_messages
from memory import MemoryGraph
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def main():
    state = State(messages=[
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="Paris"),
        HumanMessage(content="What is the capital of Germany?"),
        AIMessage(content="Berlin"),
    ], memory=MemoryGraph(nodes=[], edges=[]))
    tm = trimmed_messages(state.get("messages"), ChatOllama(model="llama3.2"))
    print(tm)

if __name__ == "__main__":
    main()
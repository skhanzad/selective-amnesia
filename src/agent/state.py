from typing import List, TypedDict, Union, Annotated
from langchain_core.messages import BaseMessage, trim_messages, HumanMessage, AIMessage
from langchain_core.messages import SystemMessage

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from memory import MemoryGraph

from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
import operator


class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    memory: MemoryGraph


def trimmed_messages(messages: List[BaseMessage], llm: Union[ChatOllama, ChatOpenAI]) -> List[BaseMessage]:
    if llm.model == "llama3.2":
        num_ctx = 131_072
    print(f"Trimming messages for {llm.model} with {num_ctx} tokens")
    
    return trim_messages(messages,
            max_tokens=num_ctx,
            strategy="last",
            token_counter="approximate",
            allow_partial=True,
            include_system=True,
        )



if __name__ == "__main__":

    llm = ChatOllama(model="llama3.2")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="Paris"),
        HumanMessage(content="What is the capital of Germany?"),
        AIMessage(content="Berlin"),
    ]
    trimmed_messages = trimmed_messages(messages, llm)
    print(trimmed_messages)
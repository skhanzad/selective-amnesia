from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from typing import Union
import os

def build_llm(model: str, provider: str = "ollama") -> Union[ChatOllama, ChatOpenAI]:
    if provider == "ollama":
        return ChatOllama(model=model)
    elif provider == "openai":
        return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    else:
        raise ValueError(f"Provider {provider} not supported")
    

if __name__ == "__main__":
    llm = build_llm(model="llama3.2", provider="ollama")
    print(llm.invoke("What is the capital of France?"))
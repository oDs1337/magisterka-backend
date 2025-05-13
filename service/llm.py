from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from .config import LLM_MODEL

class Query:
    def __init__(self, question: str, chunk: str):
        self.question = question
        self.chunk = chunk

def ask_ollama_lg(query: Query) -> str:
    prompt = PromptTemplate(
        input_variables=["question", "chunk"],
        template="Q: {question}\nContext: {chunk}\nA:"
    )
    model = OllamaLLM(model=LLM_MODEL)
    chain = prompt | model
    return chain.invoke({"question": query.question, "chunk": query.chunk})
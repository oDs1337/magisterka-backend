from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

class Query:
    def __init__(self, question, chunk):
        self.question = question
        self.chunk = chunk

def ask_ollama_lg(query: Query):
    prompt = PromptTemplate(
        input_variables=["question", "chunk"],
        template="Q: {question}\nContext: {chunk}\nA:"
    )
    model = OllamaLLM(model="llama3.2:3b")
    chain = prompt | model
    result = chain.invoke({"question": query.question, "chunk": query.chunk})
    return result
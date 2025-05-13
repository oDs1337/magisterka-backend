from .indexer import DocumentIndexer
from .llm import ask_ollama_lg, Query

indexer = DocumentIndexer()

def answer_user_question(user_question: str) -> str:
    results = indexer.query(user_question, n_results=1)
    best_chunk = results['documents'][0][0]
    query_obj = Query(user_question, best_chunk)
    return ask_ollama_lg(query_obj)
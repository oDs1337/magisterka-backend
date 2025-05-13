from service.service import chroma_collection
from service.llm import ask_ollama_lg, Query

def answer_user_question(user_question):
    results = chroma_collection.query(query_texts=user_question, n_results=1)
    best_chunk = results['documents'][0][0]
    query_obj = Query(user_question, best_chunk)
    answer = ask_ollama_lg(query_obj)
    return answer
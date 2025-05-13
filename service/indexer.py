import chromadb
from PyPDF2 import PdfReader
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from semantic_text_splitter import TextSplitter
from .config import PDF_PATH, CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

class DocumentIndexer:
    def __init__(self):
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME, embedding_function=self.embedding_function)
        self._ensure_indexed()

    def _extract_text(self) -> str:
        reader = PdfReader(PDF_PATH)
        texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
        return "".join(texts)

    def _split_text(self, text: str) -> list[str]:
        splitter = TextSplitter(1500, overlap=25)
        chunks = splitter.chunks(text)
        token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
        token_split_text = []
        for chunk in chunks:
            token_split_text += token_splitter.split_text(chunk)
        return token_split_text

    def _ensure_indexed(self):
        if not self.collection.count():
            text = self._extract_text()
            chunks = self._split_text(text)
            ids = [str(i) for i in range(len(chunks))]
            self.collection.add(ids=ids, documents=chunks)

    def query(self, query_text: str, n_results: int = 1):
        return self.collection.query(query_texts=query_text, n_results=n_results)
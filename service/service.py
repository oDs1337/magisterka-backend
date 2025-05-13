python
import chromadb
from PyPDF2 import PdfReader
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from semantic_text_splitter import TextSplitter

class DocumentIndexer:
    def __init__(
        self,
        pdf_path: str,
        chroma_db_path: str,
        collection_name: str,
        embedding_model: str,
        chunk_size: int = 1500,
        chunk_overlap: int = 25,
        tokens_per_chunk: int = 256
    ):
        self.pdf_path = pdf_path
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokens_per_chunk = tokens_per_chunk

        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        self.client = chromadb.PersistentClient(path=self.chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            self.collection_name, embedding_function=self.embedding_function
        )
        self._ensure_indexed()

    def _extract_text(self) -> str:
        reader = PdfReader(self.pdf_path)
        texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
        return "".join(texts)

    def _split_text(self, text: str) -> list[str]:
        splitter = TextSplitter(self.chunk_size, overlap=self.chunk_overlap)
        chunks = splitter.chunks(text)
        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0, tokens_per_chunk=self.tokens_per_chunk
        )
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
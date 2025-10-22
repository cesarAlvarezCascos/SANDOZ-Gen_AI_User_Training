from typing import List, Any
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from src.data_loader import load_pdf_documents

class EmbeddingPipeline:
    def __init__(self, client, model_name: str = "text-embedding-3-small", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.client = client
        self.model = client.embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        resp = self.model.create(model="text-embedding-3-small", input=texts)
        # OpenAI client returns an object with .data containing embeddings
        embs = [d.embedding for d in resp.data]
        print(f"[INFO] Generated {len(embs)} embeddings.")
        return embs


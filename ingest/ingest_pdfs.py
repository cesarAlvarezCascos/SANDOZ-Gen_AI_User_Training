import sys
import os

# Add project root to sys.path so 'src' can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from pathlib import Path
from tqdm import tqdm
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from src.data_loader import load_pdf_documents
from src.embedding import EmbeddingPipeline

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conn = psycopg.connect(os.getenv("DATABASE_URL"))

# Database helpers
def upsert_document(file_name, body):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (file_name, body)
            VALUES (%s, %s)
            RETURNING id
        """, (file_name, body))
        doc_id = cur.fetchone()[0]
    conn.commit()
    return doc_id


def upsert_chunk(document_id, idx, content):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO chunks (document_id, chunk_idx, body)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (document_id, idx, content))
        chunk_id = cur.fetchone()[0]
    conn.commit()
    return chunk_id


def insert_chunk_embedding(chunk_id, embedding):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO chunk_embeddings (chunk_id, embedding)
            VALUES (%s, %s)
        """, (chunk_id, embedding))
    conn.commit()


# Igestion pipeline
def ingest_documents(data_dir: str):

    documents = load_pdf_documents(data_dir)
    print(f"[INFO] Loaded {len(documents)} PDF documents.")

    emb_pipe = EmbeddingPipeline(client)

    chunks = emb_pipe.chunk_documents(documents)

    embeddings = emb_pipe.embed_chunks(chunks)
    if hasattr(embeddings, "data"):  
        embeddings = [d.embedding for d in embeddings.data]

    # Group chunks by file
    grouped = {}
    for chunk, emb in zip(chunks, embeddings):
        src = Path(chunk.metadata.get("source", "unknown")).name
        grouped.setdefault(src, []).append((chunk.page_content, emb))

    # Upload to database with a progress bar
    items = list(grouped.items())
    for file_name, chunk_list in tqdm(items, desc="Uploading files", unit="file"):
        tqdm.write(f"[UPLOAD] {file_name}: {len(chunk_list)} chunks")
        body = "\n".join([c[0] for c in chunk_list])
        doc_id = upsert_document(file_name, body)

        for idx, (content, emb) in enumerate(chunk_list):
            chunk_id = upsert_chunk(doc_id, idx, content)
            insert_chunk_embedding(chunk_id, emb)

        tqdm.write(f"[DONE] Uploaded {file_name}")

    print("\n Ingest complete.")



if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "pdfs")
    ingest_documents(base_dir)

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
import glob
from src.classification import init_topic_classifier_from_db, assign_topics_to_chunks


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conn = psycopg.connect(os.getenv("DATABASE_URL"))

# Database helpers
    # Insert doc in table 'documents' and return doc ID
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

    # Insert chunk in table 'chunks' and return chunk ID
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

    # Insert a chunk embedding in table 'chunk_embeddings' 
def insert_chunk_embedding(chunk_id, embedding):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO chunk_embeddings (chunk_id, embedding)
            VALUES (%s, %s)
        """, (chunk_id, embedding))
    conn.commit()


# Check duplicates: used to ingest only documents that don't exist already in the DB  

def document_exists(file_name):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM documents WHERE file_name = %s", (file_name,))
        return cur.fetchone() is not None


# Ingestion pipeline 
def ingest_documents(data_dir: str):

    pdf_files = glob.glob(os.path.join(data_dir, '**', '*.pdf'), recursive=True)
    print(f"[INFO] Loaded {len(pdf_files)} PDF documents.")

    documents = load_pdf_documents(data_dir)  # All the langchain objects/documents (1 per PDF page) concatenated in a list
    #print(f"[INFO] Loaded {len(documents)} PDF documents.") # This doesn't reflect the nº of PDF documents 

    # Group documents per PDF file and filter the duplicate ones:
    grouped_docs = {}
    skipped_files = set()
    for doc in documents:
        src = Path(doc.metadata.get("source", "unknown")).name
        if document_exists(src):
            skipped_files.add(src)
            continue  # Skip duplicated documents
        grouped_docs.setdefault(src, []).append(doc)

    # Warning about duplicates (PDFs docs with same name)
    for file_name in skipped_files:
        print(f"[SKIP] '{file_name}' already exists in database or an existing file has an identical name. Change file name if the file is not the same.")

    # Prepare list of new documents to process/load
    new_docs = [doc for docs in grouped_docs.values() for doc in docs]
    print(f"[INFO] Processing {len(grouped_docs)} new PDF files.")

    if not new_docs:
        print("[INFO] Can't find new documents to process.")
        print(f"\nIngest complete. New Files Uploaded: 0. Skipped (duplicates): {len(skipped_files)}")
        return

    # Chunking and Embeddings only for new documents
    emb_pipe = EmbeddingPipeline(client) # Initialize pipeline for generating embeddings

    chunks = emb_pipe.chunk_documents(documents)  # Separate documents into chunks 

    embeddings = emb_pipe.embed_chunks(chunks)  # Generate embedding for each chunk
    if hasattr(embeddings, "data"):  
        embeddings = [d.embedding for d in embeddings.data]  # embed_chunks ya devuelve la lista de vectores plana, así que este bloque es más para asegurar

    # Group chunks by file using a dictionary
    grouped = {}
    for chunk, emb in zip(chunks, embeddings):
        src = Path(chunk.metadata.get("source", "unknown")).name
        grouped.setdefault(src, []).append((chunk.page_content, emb))


    # Upload only new ones into the database with a progress bar
    items = list(grouped.items())
    uploaded = 0
    for file_name, chunk_list in tqdm(items, desc="Uploading files", unit="file"):
        tqdm.write(f"[UPLOAD] {file_name}: {len(chunk_list)} chunks") # How many chunks the file to be uploaded has
        body = "\n".join([c[0] for c in chunk_list])  # Joins the doc's chunks in a single body
        doc_id = upsert_document(file_name, body)

        for idx, (content, emb) in enumerate(chunk_list):
            chunk_id = upsert_chunk(doc_id, idx, content)
            insert_chunk_embedding(chunk_id, emb)

        tqdm.write(f"[DONE] Uploaded '{file_name}'")
        uploaded += 1

    print(f"\n Ingest complete. New Files Uploaded: {uploaded}. Skipped (duplicates): {len(skipped_files)}")
        # Entrenar modelo de topics (solo 1 vez tras ingestión)
    if init_topic_classifier_from_db():
        assign_topics_to_chunks(overwrite=True)



if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "pdfs")
    ingest_documents(base_dir)
import sys
import os
import hashlib
import numpy as np
import shutil

# Add project root to sys.path so 'src' can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from pathlib import Path
from tqdm import tqdm
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from src.embedding import EmbeddingPipeline
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import glob
from src.classification import init_topic_classifier_from_db, assign_topics_to_chunks

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conn = psycopg.connect(os.getenv("DATABASE_URL"))

# DOCUMENT HASH
def compute_pdf_hash(pdf_path):
    """Compute hash SHA256 of PDF binary content."""
    sha256 = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# CHECK EXISTING / DUPLICATES
def hash_exists(content_hash):
    """Verifies if an existent document already has this hash """
    with conn.cursor() as cur:
        cur.execute("SELECT id, file_name FROM documents WHERE content_hash = %s", (content_hash,))
        return cur.fetchone()
    

def archive_old_version(pdf_path):
    """Move old PDF (when it has been replaced by an updated version) to folder /pdfs/archived/"""

    archive_dir = os.path.join(os.path.dirname(pdf_path), "archived")
    os.makedirs(archive_dir, exist_ok=True)
    
    dest = os.path.join(archive_dir, os.path.basename(pdf_path))
    
    # If already existing in archived, add timestamp
    if os.path.exists(dest):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(os.path.basename(pdf_path))
        dest = os.path.join(archive_dir, f"{name}_{timestamp}{ext}")
    
    shutil.move(pdf_path, dest)
    return dest


def find_similar_document(embedding_avg, file_name, threshold=0.95):
    """
    Detects an updated version using:
    Doc. Embedding Similarity: looks for documents with similar avg embedding (> threshold)
    Filename Similarity: using pg_trm with 0.5 similarity in the filename
    Returns (doc_id, file_name, similarity) of the most similar.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                id, 
                file_name,
                1 - (embedding_avg <=> %s::vector) AS sim_embedding,
                similarity(file_name, %s) AS sim_filename
            FROM documents
            WHERE embedding_avg IS NOT NULL
            ORDER BY sim_embedding DESC
            LIMIT 1
        """, (embedding_avg, file_name))

        for doc_id, old_name, sim_emb, sim_fname in cur.fetchall():
            if sim_emb >= threshold and sim_fname > 0.5:
                return (doc_id, old_name, sim_emb)  # (id, file_name, similarity)
    
        return None
    

def delete_document_cascade(doc_id, pdf_path=None):
    """ Deletes document and all its associated chunks/embeddings"""
    with conn.cursor() as cur:
        # 1st delete chunk_embeddings
        cur.execute("""
            DELETE FROM chunk_embeddings 
            WHERE chunk_id IN (
                SELECT id FROM chunks WHERE document_id = %s
            )
        """, (doc_id,))
        
        # Then delete chunks
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        
        # Finally delete the document
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    conn.commit()

    # Delete from /pdfs
    if pdf_path and os.path.exists(pdf_path):
        archived_path = archive_old_version(pdf_path)
        tqdm.write(f"[ARCHIVE] Moved old file to: {archived_path}")


# Database helpers
    # Insert doc in table 'documents' and return doc ID
def upsert_document(file_name, body, content_hash, embedding_avg):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (file_name, body, content_hash, embedding_avg)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (file_name, body, content_hash, embedding_avg))
        doc_id = cur.fetchone()[0]
    conn.commit()
    return doc_id

    # Insert chunk in table 'chunks' and return chunk ID
def upsert_chunk(document_id, idx, content, page_number):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO chunks (document_id, chunk_idx, body, page_number)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (document_id, idx, content, page_number))
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


def sync_deletions(pdf_files):
    """
    Delete from Database documents that have been removed from /pdfs
    So if a PDF is manually deleted from the folder, it won't be used to generate answers
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id, file_name FROM documents")
        db_files = cur.fetchall()
    
    existing_names = {Path(p).name for p in pdf_files}
    
    deleted_count = 0
    for doc_id, file_name in db_files:
        if file_name not in existing_names:
            tqdm.write(f"[SYNC DELETE] Removing '{file_name}' from database (file no longer exists)")
            delete_document_cascade(doc_id)
            deleted_count += 1
    
    return deleted_count



# INGESTION PIPELINE 
def ingest_documents(data_dir: str):

    pdf_files = glob.glob(os.path.join(data_dir, '**', '*.pdf'), recursive=True)
    print(f"[INFO] Found {len(pdf_files)} PDF files in directory.")

    if not pdf_files:
        print("[INFO] No PDF files to process.")
        deleted = sync_deletions(pdf_files)
        print(f"\n{'='*60}")
        print(f"Ingestion Complete:")
        print(f"  🗑️  Files deleted from DB: {deleted}")
        print(f"{'='*60}")
        return
    
    # COUNTERS:
    uploaded = 0
    skipped_identical = 0
    updated = 0

    # Initialize pipeline
    emb_pipe = EmbeddingPipeline(client)

    # Process each doc individually
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        file_name = Path(pdf_path).name

        # 1: Compute hash of content
        content_hash = compute_pdf_hash(pdf_path)

        # 2: Verify if existing duplicate
        existing = hash_exists(content_hash)
        if existing:
            tqdm.write(f"[SKIP] '{file_name}' is exact duplicate of '{existing[1]}'")
            skipped_identical += 1
            continue

        # 3: Load and process PDF file
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to load '{file_name}': {e}")
            continue
        
        # 4: Chunking and Embeddings
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        if hasattr(embeddings, "data"):
            embeddings = [d.embedding for d in embeddings.data]

        # 5: Compute avg embedding of the doc
        embedding_avg = np.mean(embeddings, axis = 0).tolist()

        # 6: Look for a similar document
        similar = find_similar_document(embedding_avg, file_name, threshold = 0.95)

        if similar:
            # It is an UPDATED DOC
            old_id, old_name, similarity = similar
            tqdm.write(f"[UPDATE] '{file_name}' is updated version of '{old_name}' (similarity: {similarity:.3f})")

            old_pdf_path = None
            for pdf in pdf_files:
                if Path(pdf).name == old_name:
                    old_pdf_path = pdf
                    break

            tqdm.write(f"[DELETE] Removing old version '{old_name}'...")
            delete_document_cascade(old_id, old_pdf_path)
            updated += 1

        # 7: Insert New / Updated Document
        body = "\n".join([c.page_content for c in chunks])
        doc_id = upsert_document(file_name, body, content_hash, embedding_avg)

        # 8: Insert chunk and chunk_embeddings
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            page_number = chunk.metadata.get('page', None)  # Saved by langchain
            chunk_id = upsert_chunk(doc_id, idx, chunk.page_content, page_number)
            insert_chunk_embedding(chunk_id, emb)
        
        tqdm.write(f"[DONE] Uploaded '{file_name}' with {len(chunks)} chunks")
        uploaded += 1

    # 9: check for deleted files
    print("\n[INFO] Checking for deleted files...")
    deleted = sync_deletions(pdf_files)

    # 10: assign topics to chunks and propagate to documents
    print("[INFO] Initializing topics classifier...")
    init_topic_classifier_from_db()  # Train/Load Model
    
    print("[INFO] Assigning topics to chunks...")
    assign_topics_to_chunks(overwrite=True)  # Classify all chunks
    
    # Final Summary
    print(f"\n{'='*60}")
    print(f"Ingestion Complete:")
    print(f"  ✅ New files uploaded: {uploaded}")
    print(f"  🔄 Files updated: {updated}")
    print(f"  ⏭️  Exact duplicates skipped: {skipped_identical}")
    print(f"  🗑️  Files deleted from DB: {deleted}")
    print(f"{'='*60}")


if __name__ == "__main__":
    base_dir = os.getenv("PDF_FOLDER_PATH", os.path.join(os.path.dirname(__file__), "..", "pdfs"))
    ingest_documents(base_dir)

import sys
import os
import hashlib
import numpy as np
import shutil
from supabase import create_client

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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # key con permisos de escritura, para subir pdfs a Supabase Storage
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# Download from Supabase Storage to temporary local directory
# We need this because hashes, chunks and embeddings are computed in our machine, so it needs the file during that process
def download_from_supabase(file_name, temp_dir="./temp_pdfs"):
    bucket_name = "pdfs"

    # Create temp dir if not existing
    os.makedirs(temp_dir, exist_ok=True)
    local_path = os.path.join(temp_dir, file_name)

    try:
        # Download file
        res = supabase.storage.from_(bucket_name).download(file_name)

        # Save temporarly
        with open(local_path, 'wb') as f:
            f.write(res)

        return local_path

    except Exception as e:
        print(f"[ERROR] Failed to download {file_name}: {e}")
        return None



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
    

def archive_old_version(file_name):
    """Move old PDF (when it has been replaced by an updated version) to folder /pdfs/archived/ in Supabase Storage bucket"""

    bucket_name = "pdfs"

    try:
        # 1. Download actual file
        res = supabase.storage.from_(bucket_name).download(file_name)

        # 2. Upload actual file to archived/ 
        archived_path = f"archived/{file_name}"
        supabase.storage.from_(bucket_name).upload(
            path = archived_path,
            file = res,
            file_options = {"content-type": "application/pdf", "upsert": "true" }
            )
        
        # 3. Delete original file (from root pdfs/)
        supabase.storage.from_(bucket_name).remove([file_name])

        print(f"[ARCHIVED] Moved {file_name} to archived/ in bucket Storage")
        return archived_path
    
    except Exception as e:
        print(f"[ERROR] Failed to archive {file_name} in Storage: {e}")
        return None


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
    

def delete_document_cascade(doc_id, file_name=None, should_archive = False):
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
    if file_name and should_archive:
        archive_old_version(file_name)



# Database helpers
    # Insert doc in table 'documents' and return doc ID
def upsert_document(file_name, body, content_hash, embedding_avg, pdf_url):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (file_name, body, content_hash, embedding_avg, pdf_url)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (file_name, body, content_hash, embedding_avg, pdf_url))
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


def sync_deletions():
    """
    Delete from Database documents that have been removed from /pdfs (in bucket Supabase Storage)
    So if a PDF is manually deleted from the folder, it won't be used to generate answers
    """
    bucket_name = "pdfs"

    # List files in Storage pdfs/
    try:
        storage_files = supabase.storage.from_(bucket_name).list()
        # Only in root pdfs/ (not in pdfs/archived)
        existing_names = {
            f['name'] for f in storage_files 
            if f['name'].endswith('.pdf') and not f['name'].startswith('archived/')
        }
    except Exception as e:
        print(f"[ERROR] Could not list Storage files: {e}")
        return 0


    with conn.cursor() as cur:
        cur.execute("SELECT id, file_name FROM documents")
        db_files = cur.fetchall()
    
    
    deleted_count = 0
    for doc_id, file_name in db_files:
        if file_name not in existing_names:
            tqdm.write(f"[SYNC DELETE] Removing '{file_name}' from database (file no longer exists)")
            delete_document_cascade(doc_id, file_name)
            deleted_count += 1
    
    return deleted_count



# INGESTION PIPELINE (reading PDFs files from Supabase Storage)
def ingest_documents():

    bucket_name = "pdfs"
    # List files in bucket Storage
    try:
        storage_files = supabase.storage.from_(bucket_name).list()
        # Filter only those in root /pdfs (not in archived/)
        pdf_files = [
            f['name'] for f in storage_files
            if f['name'].endswith('.pdf') and not f['name'].startswith('archived/')
        ]

    except Exception as e:
        print(f"[ERROR] Could not list Storage: {e}")
        return

    print(f"[INFO] Found {len(pdf_files)} PDF files in Storage bucket '{bucket_name}.")

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

    # Create temp directory
    temp_dir = "./temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)

    # Process each doc individually
    for file_name in tqdm(pdf_files, desc="Processing PDFs", unit="file"):

        # Download PDF temporarily, so we can compute hashes, embeddings, etc (our machine needs those files)
        pdf_path = download_from_supabase(file_name, temp_dir)
        if not pdf_path:
            tqdm.write(f"[ERROR] Could not download '{file_name}'")
            continue


        # 1: Compute hash of content
        content_hash = compute_pdf_hash(pdf_path)

        # 2: Verify if existing duplicate
        existing = hash_exists(content_hash)
        if existing:
            tqdm.write(f"[SKIP] '{file_name}' is exact duplicate of '{existing[1]}'")
            skipped_identical += 1
            os.remove(pdf_path)  # Clean temp
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
            tqdm.write(f"[DELETE] Removing old version '{old_name}'...")
            delete_document_cascade(old_id, old_name, should_archive=True)
            updated += 1


        # 7: Insert New / Updated Document
        body = "\n".join([c.page_content for c in chunks])
            # URL that points to the document in the bucket Storage
        pdf_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{file_name}"
        doc_id = upsert_document(file_name, body, content_hash, embedding_avg, pdf_url)

        # 8: Insert chunk and chunk_embeddings
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            page_number = chunk.metadata.get('page', None)  # Saved by langchain
            chunk_id = upsert_chunk(doc_id, idx, chunk.page_content, page_number)
            insert_chunk_embedding(chunk_id, emb)
        
        tqdm.write(f"[DONE] Uploaded '{file_name}' with {len(chunks)} chunks")
        uploaded += 1

        # Clean temp dir
        os.remove(pdf_path)

    # 9: check for deleted files
    print("\n[INFO] Checking for deleted files...")
    deleted = sync_deletions()

    # 10: Assign topics to chunks and propagate to documents
    print("[INFO] Initializing topics classifier...")
    init_topic_classifier_from_db()  # Train/Load Model
    
    print("[INFO] Assigning topics to chunks...")
    # assign_topics_to_chunks(overwrite=True)  # Classify all chunks
    assign_topics_to_chunks(overwrite=False)  # False because we dont want to reassign already assigned topic to already ingested chunks
    
    # Final Summary
    print(f"\n{'='*60}")
    print(f"Ingestion Complete:")
    print(f"  ✅ New files uploaded: {uploaded}")
    print(f"  🔄 Files updated: {updated}")
    print(f"  ⏭️  Exact duplicates skipped: {skipped_identical}")
    print(f"  🗑️  Files deleted from DB: {deleted}")
    print(f"{'='*60}")

    # Clean temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    ingest_documents()  # We don't need base_dir as before

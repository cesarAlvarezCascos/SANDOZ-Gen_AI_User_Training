import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
# Add project root to sys.path so 'src' can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.embedding import EmbeddingPipeline
from src.classification import init_topic_classifier_from_db, assign_topics_to_chunks
from ingest.DBops import DatabaseOperator
from ingest.docling_loader import CustomDocumentLoader

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


# INGESTION PIPELINE 
def ingest_documents(data_dir: str):

    operator = DatabaseOperator(SUPABASE_URL, SUPABASE_KEY, data_dir)
    print(f"[INFO] Found {len(operator.pdf_files)} PDF files in directory.")

    if not operator.pdf_files:
        print("[INFO] No PDF files to process.")
        deleted = operator.sync_deletions()
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
    for pdf_path in tqdm(operator.pdf_files, desc="Processing PDFs", unit="file"):
        file_name = Path(pdf_path).name
        # 1: Compute hash of content and verify if existing duplicate
        content_hash = operator.compute_pdf_hash(pdf_path)
        existing = operator.check_hash(content_hash)
        if existing:
            tqdm.write(f"[SKIP] '{file_name}' is exact duplicate of '{existing[1]}'")
            skipped_identical += 1
            continue

        # 2: Load and process PDF file
        try:
            loader = CustomDocumentLoader(file_path=pdf_path)
            chunks = loader.load()
        except Exception as e:
            tqdm.write(f"[ERROR] Docling failed to load '{file_name}': {e}")
            continue
        
        # 4: Embeddings
        embeddings = emb_pipe.embed_chunks(chunks)
        if hasattr(embeddings, "data"):
            embeddings = [d.embedding for d in embeddings.data]

        # 5: Compute avg embedding of the doc
        embedding_avg = np.mean(embeddings, axis = 0).tolist()

        # 6: Look for a similar document
        similar = operator.find_similar_document(embedding_avg, pdf_path, threshold = 0.95)

        if similar:
            # It is an UPDATED DOC
            old_id, old_name, similarity = similar
            tqdm.write(f"[UPDATE] '{file_name}' is updated version of '{old_name}' (similarity: {similarity:.3f})")

            old_pdf_path = None
            for pdf in operator.pdf_files:
                if Path(pdf).name == old_name:
                    old_pdf_path = pdf
                    break

            tqdm.write(f"[DELETE] Removing old version '{old_name}'...")
            operator.delete_document_cascade(old_id, old_pdf_path)
            updated += 1

        # 7: Insert New / Updated Document
        body = "\n".join([c.page_content for c in chunks])
        pdf_url = operator.upload_pdf(pdf_path)  # generate URL for that document stored in our Supabase storage bucket 
        doc_id = operator.upsert_document(file_name, body, content_hash, embedding_avg, pdf_url)

        # 8: Insert chunk and chunk_embeddings
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            page_number = chunk.metadata.get('page', None)  # Saved by langchain
            chunk_id = operator.upsert_chunk(doc_id, idx, chunk.page_content, page_number)
            operator.insert_chunk_embedding(chunk_id, emb)
        
        tqdm.write(f"[DONE] Uploaded '{file_name}' with {len(chunks)} chunks")
        uploaded += 1

        operator.conn.commit()

    # 9: check for deleted files
    print("\n[INFO] Checking for deleted files...")
    deleted = operator.sync_deletions()

    # 10: assign topics to chunks and propagate to documents
    print("[INFO] Initializing topics classifier...")
    init_topic_classifier_from_db()  # Train/Load Model
    
    print("[INFO] Assigning topics to chunks...")
    # assign_topics_to_chunks(overwrite=True)  # Classify all chunks
    assign_topics_to_chunks(overwrite=False)
    
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

import supabase
import os
from dotenv import load_dotenv
import psycopg
import hashlib
import shutil
import tqdm
from pathlib import Path
import glob

class DatabaseOperator():
    def __init__(self, url: str, key: str, path: str):
        load_dotenv()
        self.conn = psycopg.connect(os.getenv("DATABASE_URL"))
        self.client = supabase.create_client(url, key)
        self.data_dir = path
        self.bucket = "pdfs"
        self.pdf_files = self._get_pdf_files(self.data_dir)
    
    
    def upload_pdf(self, file_path: str):
        """Upload PDF to Supabase Storage and return public URL."""
        file_name = os.path.basename(file_path)
        
        with open(file_path, "rb") as f:
            data = f.read()

        # Overwrite if existing
        try: 
            res = supabase.storage.from_(self.bucket).upload(
                path = file_name, 
                file = data, 
                file_options = {"cache-control": "3600", "upsert": "true", "content-type": "application/pdf"}
                )
        
        except Exception as e:
            print(f"[ERROR] Failed to upload {file_name} to Supabase: {e}")
            return None

        # Generate public or signed URL
        try:
            # url = supabase.storage.from_(bucket_name).get_public_url(file_name).get("publicUrl")
            url = supabase.storage.from_(self.bucket).get_public_url(file_name)
            return url
        except Exception as e:
            print(f"[ERROR] Failed to get public URL for {file_name}: {e}")
            return None 
        

    def delete_pdf(self, file_path: str):
        """Delete PDF from Supabase Storage."""
        file_name = os.path.basename(file_path)
        if file_name:
            res = self.client.storage.from_(self.bucket).remove([file_name])
            if res.get("error"):
                print(f"[ERROR] Could not delete {file_name} from Supabase: {res['error']}")
            else:
                print(f"[DELETE] {file_name} removed from Supabase")


    def compute_pdf_hash(self, file_path: str) -> str:
        """Compute hash SHA256 of PDF binary content."""

        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    

    def hash_exists(self, file_path: str):
        """Verifies if an existent document already has this hash """

        content_hash = self.compute_pdf_hash(file_path)
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, file_name " \
            "FROM documents " \
            "WHERE content_hash = %s", (content_hash,))
            return cur.fetchone()
        
        
    def archive_old_version(self, file_path: str) -> str:
        """Move old PDF (when it has been replaced by an updated version) 
        to folder /pdfs/archived/"""

        archive_dir = os.path.join(os.path.dirname(file_path), "archived")
        os.makedirs(archive_dir, exist_ok=True)
        
        dest = os.path.join(archive_dir, os.path.basename(file_path))
        
        # If already existing in archived, add timestamp
        if os.path.exists(dest):
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(os.path.basename(file_path))
            dest = os.path.join(archive_dir, f"{name}_{timestamp}{ext}")
        
        shutil.move(file_path, dest)
        return dest
    

    def find_similar_document(self, embedding_avg, file_path: str, threshold=0.95):
        """
        Detects an updated version using:
        Doc. Embedding Similarity: looks for documents with similar avg embedding (> threshold)
        Filename Similarity: using pg_trm with 0.5 similarity in the filename
        Returns (doc_id, file_name, similarity) of the most similar.
        """
        file_name = os.path.basename(file_path)
        with self.conn.cursor() as cur:
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
        
    
    def delete_document_cascade(self, doc_id, file_path: str = None):
        """ Deletes document and all its associated chunks/embeddings"""
        with self.conn.cursor() as cur:
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
        self.conn.commit()

        # Delete from /pdfs
        if file_path and os.path.exists(file_path):
            archived_path = self.archive_old_version(file_path)
            tqdm.write(f"[ARCHIVE] Moved old file to: {archived_path}")

        # Delete from Supabase Storage
        if file_path:
            self.delete_pdf(os.path.basename(file_path))

    
    def upsert_document(self, file_name, body, content_hash, embedding_avg, pdf_url):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (file_name, body, content_hash, embedding_avg, pdf_url)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (file_name, body, content_hash, embedding_avg, pdf_url))
            doc_id = cur.fetchone()[0]
        self.conn.commit()
        return doc_id


    def upsert_chunk(self, document_id, idx, content, page_number):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunks (document_id, chunk_idx, body, page_number)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (document_id, idx, content, page_number))
            chunk_id = cur.fetchone()[0]
        self.conn.commit()
        return chunk_id


    def insert_chunk_embedding(self, chunk_id, embedding):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunk_embeddings (chunk_id, embedding)
                VALUES (%s, %s)
            """, (chunk_id, embedding))
        self.conn.commit()

    
    def sync_deletions(self):
        """
        Delete from Database documents that have been removed from /pdfs
        So if a PDF is manually deleted from the folder, it won't be used to generate answers
        """

        with self.conn.cursor() as cur:
            cur.execute("SELECT id, file_name FROM documents")
            db_files = cur.fetchall()
        
        existing_names = {Path(p).name for p in self.pdf_files}
        
        deleted_count = 0
        for doc_id, file_name in db_files:
            if file_name not in existing_names:
                tqdm.write(f"[SYNC DELETE] Removing '{file_name}' from database (file no longer exists)")
                self.delete_document_cascade(doc_id)
                deleted_count += 1
        
        return deleted_count
    
    @staticmethod
    def _get_pdf_files(path: str):
        """Get list of all PDF files in a directory (recursively)."""
        return glob.glob(os.path.join(path, '**', '*.pdf'), recursive=True)
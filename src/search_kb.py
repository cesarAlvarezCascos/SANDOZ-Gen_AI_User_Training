import os, psycopg
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conn = psycopg.connect(os.getenv("DATABASE_URL"))


def _embed(q: str):
    """Create a 1536-dim embedding vector from the user query."""
    return client.embeddings.create(model="text-embedding-3-small", input=q).data[0].embedding


def vector_search(qvec, k=12):
    """
    Perform similarity search using cosine distance on chunk embeddings.
    Joins `chunk_embeddings` -> `chunks` -> `documents`.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                c.id AS chunk_id,
                c.body AS chunk_text,
                d.id AS document_id,
                d.file_name,
                c.page_number,
                d.pdf_url,
                1 - (ce.embedding <=> %s::vector) AS vscore
            FROM chunk_embeddings ce
            JOIN chunks c ON ce.chunk_id = c.id
            JOIN documents d ON c.document_id = d.id
            ORDER BY ce.embedding <=> %s::vector
            LIMIT %s;
        """, (qvec, qvec, k))
        return cur.fetchall()


def keyword_search(query, k=12):
    """
    Perform keyword-based search using pg_trgm similarity between query text 
    and chunks. Returns the top-k matches (most relevant chunks).
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                c.id AS chunk_id,
                c.body AS chunk_text,
                d.id AS document_id,
                d.file_name,
                c.page_number,
                d.pdf_url,
                similarity(c.body, %s) AS kscore
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE similarity(c.body, %s) > 0.1
            ORDER BY kscore DESC
            LIMIT %s;
        """, (query, query, k))
        return cur.fetchall()
    


def fuse(vec_hits, key_hits, alpha=0.6, top=8):
    """
    Combine vector search and keyword search scores using a weighted fusion.
    alpha controls the weight given to vector similarity.
    """
    S = defaultdict(float) # Weighted sum of scores per chunk
    M = {}  # chunk snippet (txt), source doc (filename), doc id (did) and page number (page_num) for each chunk

    for (cid, txt, did, filename, page_num, pdf_url, sc) in vec_hits:
        if cid not in M:
            M[cid] = (txt, filename, did, page_num, pdf_url)
        S[cid] += alpha * float(sc)

    for (cid, txt, did, filename, page_num, pdf_url, sc) in key_hits:
        if cid not in M:
            M[cid] = (txt, filename, did, page_num, pdf_url)
        S[cid] += (1 - alpha) * float(sc)

    ranked = sorted(S.items(), key=lambda x: x[1], reverse=True)[:top]

    # Return the info we want in the Citations JSON
    return [
        {
            # Ensure values are JSON-serializable (UUIDs -> str)
            "chunk_id": str(cid),
            "file_name": str(M[cid][1]),
            "document_id": str(M[cid][2]),
            "page_number": (M[cid][3] + 1) if M[cid][3] is not None else None,
            "pdf_url": M[cid][4],
            "snippet": M[cid][0][:600],
            "score": float(s)
        }
        for cid, s in ranked
    ]


def search_kb(query: str, top_k: int = 8):
    """
    1. Embed query → vector search
    2. Keyword search → fuse both
    """
    qvec = _embed(query)
    vec_hits = vector_search(qvec, k=top_k * 2)
    key_hits = keyword_search(query, k=top_k * 2)
    results = fuse(vec_hits, key_hits, alpha=0.6, top=top_k)
    return results
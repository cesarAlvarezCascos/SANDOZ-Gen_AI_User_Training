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
                d.source_path,
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
    and document body/summary. Returns the top-k matches.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                c.id AS chunk_id,
                c.body AS chunk_text,
                d.id AS document_id,
                d.source_path,
                GREATEST(similarity(d.body, %s), similarity(d.summary, %s)) AS kscore
            FROM documents d
            JOIN chunks c ON c.document_id = d.id
            ORDER BY kscore DESC
            LIMIT %s;
        """, (query, query, k))
        return cur.fetchall()


def fuse(vec_hits, key_hits, alpha=0.6, top=8):
    """
    Combine vector search and keyword search scores using a weighted fusion.
    alpha controls the weight given to vector similarity.
    """
    S = defaultdict(float)
    M = {}

    for (_id, txt, did, url, sc) in vec_hits:
        if did not in M:
            M[did] = (txt, url)
        S[did] += alpha * float(sc)

    for (_id, txt, did, url, sc) in key_hits:
        if did not in M:
            M[did] = (txt, url)
        S[did] += (1 - alpha) * float(sc)

    ranked = sorted(S.items(), key=lambda x: x[1], reverse=True)[:top]
    return [
        {
            "document_id": mid,
            "snippet": M[mid][0][:600],
            "source_path": M[mid][1],
            "score": float(s)
        }
        for mid, s in ranked
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
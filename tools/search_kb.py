import os, psycopg
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conn = psycopg.connect(os.getenv("DATABASE_URL"))

def _embed(q:str):
    return client.embeddings.create(model="text-embedding-3-small", input=q).data[0].embedding

def vector_search(qvec, role, k=12):
    with conn.cursor() as cur:
        cur.execute("""
          SELECT c.id, c.text, m.id, m.source_url,
                 1 - (c.embedding <=> %s::vector) AS vscore
          FROM material_chunks c
          JOIN materials m ON m.id = c.material_id
          LEFT JOIN access_policies ap ON ap.material_id = m.id
          WHERE (ap.classification IS NULL OR ap.classification != 'restricted' OR %s = ANY(ap.allowed_roles))
          ORDER BY c.embedding <=> %s::vector
          LIMIT %s
        """, (qvec, role, qvec, k))
        return cur.fetchall()

def keyword_search(query, role, k=12):
    with conn.cursor() as cur:
        cur.execute("""
          SELECT c.id, c.text, m.id, m.source_url,
                 GREATEST(similarity(m.body, %s), similarity(m.summary, %s)) AS kscore
          FROM materials m
          JOIN material_chunks c ON c.material_id = m.id
          LEFT JOIN access_policies ap ON ap.material_id = m.id
          WHERE (ap.classification IS NULL OR ap.classification != 'restricted' OR %s = ANY(ap.allowed_roles))
          ORDER BY kscore DESC
          LIMIT %s
        """, (query, query, role, k))
        return cur.fetchall()

def fuse(vec_hits, key_hits, alpha=0.6, top=8):
    from collections import defaultdict
    S = defaultdict(float); M = {}
    for (_id, txt, mid, url, sc) in vec_hits:
        if mid not in M: M[mid] = (txt, url)
        S[mid] += alpha*float(sc)
    for (_id, txt, mid, url, sc) in key_hits:
        if mid not in M: M[mid] = (txt, url)
        S[mid] += (1-alpha)*float(sc)
    ranked = sorted(S.items(), key=lambda x: x[1], reverse=True)[:top]
    return [{"material_id": mid, "snippet": M[mid][0][:600], "source_url": M[mid][1], "score": float(s)} for mid,s in ranked]

def search_kb(query:str, role:str="analyst"):
    qvec = _embed(query)
    v = vector_search(qvec, role, k=12)
    k = keyword_search(query, role, k=12)
    return fuse(v, k, alpha=0.6, top=8)

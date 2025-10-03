import os, re, psycopg
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conn = psycopg.connect(os.getenv("DATABASE_URL"))

CHUNK_TOKENS = 350
OVERLAP = 60

def pdf_to_text(path):
    r = PdfReader(path)
    text = "\n".join((p.extract_text() or "") for p in r.pages)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def chunk(text, n=CHUNK_TOKENS, overlap=OVERLAP):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, buf, tok = [], [], 0
    for p in paras:
        w = p.split()
        if tok + len(w) > n and buf:
            out.append(" ".join(buf))
            buf = buf[-overlap:]; tok = len(buf)
        buf += w; tok += len(w)
    if buf: out.append(" ".join(buf))
    return out

def embed(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

def upsert_material(title, body, source_url):
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO materials(title, summary, body, modality, source_url)
          VALUES (%s,%s,%s,'doc',%s) RETURNING id
        """, (title, body[:500], body, source_url))
        mid = cur.fetchone()[0]
    conn.commit()
    return mid

def upsert_chunks(material_id, chunks, embs):
    with conn.cursor() as cur:
        for i,(t,e) in enumerate(zip(chunks, embs)):
            cur.execute("""
              INSERT INTO material_chunks(material_id,chunk_idx,text,embedding)
              VALUES (%s,%s,%s,%s)
            """, (material_id, i, t, e))
    conn.commit()

if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pdfs"))
    pdfs = [
        "Introduction to Pre-Selection Pipeline.pdf",
        "Understanding IP Information.pdf",
        "CCO relevant Abbreviations & Definitions.pdf",
        "sandoz_agents_for_userTraining.pdf",
    ]
    for name in pdfs:
        path = os.path.join(base, name)
        body = pdf_to_text(path)
        mid = upsert_material(name, body, f"file://{path}")
        chunks = chunk(body)
        embs = embed(chunks)
        upsert_chunks(mid, chunks, embs)
    print("Ingesta completa")

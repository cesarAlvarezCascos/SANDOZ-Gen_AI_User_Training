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


def vector_search(qvec, k=12, topic_name=None):
    """
    Perform similarity search using cosine distance on chunk embeddings.
    Joins `chunk_embeddings` -> `chunks` -> `documents`.
    """
    with conn.cursor() as cur:
        if topic_name: #added 
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
                WHERE c.topic_id = %s
                ORDER BY ce.embedding <=> %s::vector
                LIMIT %s;
            """, (qvec, topic_name, qvec, k))
        else:
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



def keyword_search(query, k=12, topic_id=None):
    """
    Perform keyword-based search using pg_trgm similarity between query text 
    and chunks. If a topic_id is provided, restrict search to that topic.
    """
    with conn.cursor() as cur:
        if topic_id:
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
                WHERE c.topic_id = %s
                AND similarity(c.body, %s) > 0.1
                ORDER BY kscore DESC
                LIMIT %s;
            """, (query, topic_id, query, k))
        else:
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
    


def fuse(vec_hits, key_hits, alpha=0.6, top=8, relevance_threshold=0.1):
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

    # Apply relevance threshold
    S = {cid: score for cid, score in S.items() if score >= relevance_threshold}
    if len(S) == 0:
        return []

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


def detect_topic(query: str):
    """
    Checks if the user's question contains any keyword from the 'topics.key_words' column.
    Returns (id, name) of the most relevant topic if matched, or None if there is no match.
    Returns a list of topics 
    """
    with conn.cursor() as cur:
        # We only select the columns we need
        cur.execute("SELECT id, name, key_words FROM topics;")
        topics = cur.fetchall()  # Each row: (id, name, [keywords])

    query_lower = query.lower()
    matched_topic = None
    best_match_count = 0

    for tid, name, keywords in topics:
        if not keywords:
            continue

        # Count how many keywords appear in the query
        match_count = sum(1 for kw in keywords if kw.lower() in query_lower)

        # Keep the topic with the highest number of matches
        if match_count > best_match_count:
            best_match_count = match_count
            matched_topic = (tid, name)

    # if no topic detected, all are return so the user chooses
    if best_match_count == 0:
        all_topics = [{"name": name} for _, name, _ in topics]
        return None, all_topics

    return matched_topic, []


def search_kb(query: str, top_k: int = 8, selected_topic_name: str | None = None):
    """
    0. Detect topic (if any)
    1. Embed query → vector search
    2. Keyword search → fuse both
    """
    if selected_topic_name:
        # the user selects manually a topic because detect_topic fails
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM topics WHERE name = %s", (selected_topic_name,))
            row = cur.fetchone()
            topic_id = row[0] if row else None
            topic_name = selected_topic_name if row else None
    else:
        # try to detect topic using detect_topic
        matched_topic, all_topics = detect_topic(query)
        if matched_topic is None:
            # if no topic selected → frontend must ask
            return {
                "status": "choose_topic",
                "message": "A topic has not been detected. Please, select one of the following topics.",
                "topics": all_topics
            }
        topic_id, topic_name = matched_topic

    qvec = _embed(query)
    vec_hits = vector_search(qvec, k=top_k * 2, topic_name=topic_name)
    key_hits = keyword_search(query, k=top_k * 2, topic_id=topic_id)
    results = fuse(vec_hits, key_hits, alpha=0.6, top=top_k)
    return results

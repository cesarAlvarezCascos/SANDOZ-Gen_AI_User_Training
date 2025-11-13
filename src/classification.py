# src/classification.py
import os
import psycopg
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import spacy
from joblib import dump, load

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conn = psycopg.connect(os.getenv("DATABASE_URL"))
MODEL_PATH = "models/topic_model.joblib"
conn.autocommit = True


# Cargar modelo spaCy
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    raise RuntimeError(
        "[ERROR] No se encontró el modelo spaCy 'en_core_web_lg'. "
        "Instálalo con: python -m spacy download en_core_web_lg"
    )

# # Etiquetas de topics (puedes ajustarlas según tus PDFs)
# TOPIC_LABELS = {
#     0: "Wizard's Country Sales Forecast Record",
#     1: "PSG Document Evaluation Opportunities",
#     2: "Car Deal Approval: Partner Manager",
#     3: "Sandoz Product Portfolio: Target and API Origins",
#     4: "Creating Meeting Agenda and Record Selection",
#     5: "Proper User Access to AI in Countries",
#     6: "Risk of IP Exclusivity Post-Launch",
#     7: "Regional Business Portfolio Forecast Request",
#     8: "Medical Team Evaluation and Opportunities"
# }

# Globals para modelo y vectorizer
_VECTORIZER = None
_NMF_MODEL = None

def _get_topic_label(topic_idx: int) -> str:
    """
    Devuelve el nombre del topic a partir de topics_test
    usando model_topic_index. Si no existe, devuelve 'Topic N'.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(name_manual, name)
            FROM topics_test
            WHERE model_topic_index = %s
            ORDER BY id
            LIMIT 1;
            """,
            (topic_idx,)
        )
        row = cur.fetchone()
        if row and row[0]:
            return row[0]

    return f"Topic {topic_idx+1}"


def generate_topic_metadata_with_llm(top_keywords, sample_snippets):
    """
    Usa un LLM para generar:
    - title: nombre corto del topic
    - description: descripción breve del cluster
    """
    # Limitamos un poco el tamaño de los snippets
    joined_snippets = "\n---\n".join(s[:400] for s in sample_snippets[:15])

    prompt = f"""
You are labeling topics for an internal knowledge base.

You receive:
- A list of representative keywords of one topic.
- Several text snippets that belong to this topic.

Your task:
1) Propose a SHORT, clear title for this topic (max 7 words).
2) Write a BRIEF description (1-2 sentences, max 50 words)
   explaining what kind of documents this topic groups and what they are about.

Return ONLY a JSON object with this exact structure:

{{
  "title": "...",
  "description": "..."
}}

Do not add any commentary, explanations or extra text.

Keywords:
{", ".join(top_keywords)}

Snippets:
{joined_snippets}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You ONLY return valid JSON. No explanations, no markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0.3,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()

        import json
        data = json.loads(raw)

        title = data.get("title", "").strip()
        description = data.get("description", "").strip()

        if not title:
            title = "Untitled topic"
        if not description:
            description = "Cluster of related documents for this topic."

        return title, description

    except Exception as e:
        print(f"[WARN] LLM topic metadata generation failed: {e}")
        # Fallback simple
        title = ", ".join(top_keywords[:4])
        description = "Cluster of documents related to: " + ", ".join(top_keywords[:6])
        return title, description


# Funciones utilitarias
def preprocess(text: str) -> str:
    """Lematiza y elimina stopwords y símbolos no alfabéticos."""
    if not text:
        return ""
    doc = nlp(text)
    return " ".join([t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha])

def train_topic_model(texts, n_components: int = 9, max_features: int = 1000):
    """Entrena TF-IDF + NMF sobre una lista de textos."""
    processed = [preprocess(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(processed)
    nmf_model = NMF(n_components=n_components, random_state=42)
    nmf_model.fit(X)
    print(f"[INFO] Topic model entrenado con {n_components} componentes.")
    return nmf_model, vectorizer

# Inicializador global desde la base de datos
def init_topic_classifier_from_db(n_components: int = 9, max_features: int = 1000, limit: int | None = None):
    """Carga los textos de los chunks y entrena el modelo de topics."""
    global _NMF_MODEL, _VECTORIZER
    # Si ya existe el modelo entrenado, lo cargamos directamente
    if os.path.exists(MODEL_PATH):
        try:
            _NMF_MODEL, _VECTORIZER = load(MODEL_PATH)
            print(f"[INFO] Modelo de tópicos cargado desde '{MODEL_PATH}'.")
            return True
        except Exception as e:
            print(f"[WARN] No se pudo cargar el modelo guardado: {e}. Se entrenará de nuevo.")

    with conn.cursor() as cur:
        if limit:
            cur.execute("SELECT body FROM chunks ORDER BY id LIMIT %s;", (limit,))
        else:
            cur.execute("SELECT body FROM chunks ORDER BY id;")
        texts = [r[0] for r in cur.fetchall() if r[0]]

    if not texts:
        raise RuntimeError("[ERROR] No hay chunks disponibles para entrenar el clasificador de topics.")

    _NMF_MODEL, _VECTORIZER = train_topic_model(texts, n_components, max_features)
    # Guardar modelo entrenado en disco
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump((_NMF_MODEL, _VECTORIZER), MODEL_PATH)
    print(f"[INFO] Topic classifier initialized with {len(texts)} chunks.")
    return True

def build_test_topics_from_model(n_top_words: int = 10, n_sample_chunks: int = 15):
    """
    Construye la tabla topics_test a partir del modelo NMF entrenado:
    - Saca las top keywords de cada componente (topic).
    - Elige algunos chunks representativos por topic.
    - Llama a un LLM para generar name + descripción.
    - Inserta name + key_words + descripcion en topics_test.
    """
    global _NMF_MODEL, _VECTORIZER
    if _NMF_MODEL is None or _VECTORIZER is None:
        raise RuntimeError("[ERROR] init_topic_classifier_from_db() must be called first.")

    feature_names = _VECTORIZER.get_feature_names_out()
    n_components = _NMF_MODEL.n_components

    # 1) Cogemos todos los textos de chunks para poder elegir ejemplos por topic
    with conn.cursor() as cur:
        cur.execute("SELECT id, body FROM chunks ORDER BY id;")
        rows = cur.fetchall()

    if not rows:
        print("[WARN] No chunks found when building topics_test.")
        return

    chunk_ids = [r[0] for r in rows]
    texts_raw = [r[1] or "" for r in rows]

    # Preprocesar como en entrenamiento
    processed_texts = [preprocess(t) for t in texts_raw]
    X = _VECTORIZER.transform(processed_texts)
    # Matriz documento x topic
    W = _NMF_MODEL.transform(X)

    topics_payload = []

    for k in range(n_components):
        # 2) Top palabras del componente k
        comp_weights = _NMF_MODEL.components_[k]
        top_idx = comp_weights.argsort()[::-1][:n_top_words]
        top_keywords = [feature_names[i] for i in top_idx]

        # 3) Elegir algunos chunks representativos (los que más peso tienen en este topic)
        topic_weights = W[:, k]
        if (topic_weights > 0).sum() == 0:
            sample_snippets = []
        else:
            sample_idx = topic_weights.argsort()[::-1][:n_sample_chunks]
            sample_snippets = [texts_raw[i] for i in sample_idx]

        # 4) Llamar al LLM para generar nombre + descripción
        title, description = generate_topic_metadata_with_llm(top_keywords, sample_snippets)

        topics_payload.append((k, title, description, top_keywords))

        print(f"[TEST TOPIC {k}] {title}")
        print(f"   keywords: {', '.join(top_keywords[:6])}")
        print(f"   desc: {description[:80]}...")

    # 5) Volcar en topics_test
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE topics_test;")
        for model_idx, name, desc, kws in topics_payload:
            cur.execute(
                """
                INSERT INTO topics_test
                    (model_topic_index, name, key_words, descripcion, is_active)
                VALUES (%s, %s, %s, %s, TRUE);
                """,
                (model_idx, name, kws, desc)
            )
    conn.commit()
    print(f"[INFO] Created {len(topics_payload)} test topics in 'topics_test'.")


# Clasificación individual
def classify_topic(text: str) -> str:
    """Clasifica un texto corto en uno de los topics conocidos."""
    global _NMF_MODEL, _VECTORIZER
    if _NMF_MODEL is None or _VECTORIZER is None:
        raise RuntimeError("[ERROR] Call init_topic_classifier_from_db() first.")

    processed = preprocess(text)
    X = _VECTORIZER.transform([processed])
    topic_idx = int(np.argmax(_NMF_MODEL.transform(X)))
    return _get_topic_label(topic_idx)

# Asignación masiva de topics a chunks
def assign_topics_to_chunks(limit: int | None = None, overwrite: bool = True, batch_size: int = 50, n_process: int = 2):
    """Asigna un topic a cada chunk según el modelo entrenado."""
    if _NMF_MODEL is None or _VECTORIZER is None:
        raise RuntimeError("Inicializa el clasificador con init_topic_classifier_from_db() antes.")

    with conn.cursor() as cur:
        if overwrite:
            q = "SELECT id, body FROM chunks ORDER BY id"
        else:
            q = "SELECT id, body FROM chunks WHERE topic_id IS NULL ORDER BY id"
        if limit:
            q += f" LIMIT {limit}"
        cur.execute(q)
        chunks = cur.fetchall()

    if not chunks:
        print("[INFO] No chunks to classify.")
        return

    print(f"[INFO] Classifying {len(chunks)} chunks (overwrite={overwrite})...")

    chunk_ids = [r[0] for r in chunks]
    texts = [r[1] for r in chunks]
    docs = list(nlp.pipe(texts, batch_size=batch_size, n_process=n_process))
    processed_texts = [
        " ".join([t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha])
        for doc in docs
    ]

    X = _VECTORIZER.transform(processed_texts)
    topic_indices = np.argmax(_NMF_MODEL.transform(X), axis=1)

    with conn.cursor() as cur:
        for chunk_id, topic_idx in zip(chunk_ids, topic_indices):
            topic_name = _get_topic_label(topic_idx)
            # Insertar o recuperar topic_id
            cur.execute("SELECT id FROM topics WHERE name = %s;", (topic_name,))
            row = cur.fetchone()
            if row:
                topic_id = row[0]
            else:
                cur.execute("INSERT INTO topics (name) VALUES (%s) RETURNING id;", (topic_name,))
                topic_id = cur.fetchone()[0]
            cur.execute("UPDATE chunks SET topic_id = %s WHERE id = %s;", (topic_id, chunk_id))

        try:
            cur.execute("""
                UPDATE documents d
                SET topic = sub.topic
                FROM (
                    SELECT c.document_id, ARRAY_AGG(DISTINCT t.name) AS topic
                    FROM chunks c
                    JOIN topics t ON c.topic_id = t.id
                    GROUP BY c.document_id
                ) AS sub
                WHERE d.id = sub.document_id;
            """)
        except psycopg.errors.UndefinedColumn:
            print("[WARN] 'documents.topics' does not exist, update is omitted.")

    print("[INFO] Topics reassignation completed.")

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "topics_test"

    if mode == "topics_test":
        # 1) Entrenamos / cargamos modelo de topics
        init_topic_classifier_from_db()
        # 2) Generamos topics_test usando LLM para nombrar
        build_test_topics_from_model()
    else:
        print(f"[INFO] Unknown mode '{mode}'. Use: python -m src.classification topics_test")

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

# Etiquetas de topics (puedes ajustarlas según tus PDFs)
TOPIC_LABELS = {
    0: "Wizard's Country Sales Forecast Record",
    1: "PSG Document Evaluation Opportunities",
    2: "Car Deal Approval: Partner Manager",
    3: "Sandoz Product Portfolio: Target and API Origins",
    4: "Creating Meeting Agenda and Record Selection",
    5: "Proper User Access to AI in Countries",
    6: "Risk of IP Exclusivity Post-Launch",
    7: "Regional Business Portfolio Forecast Request",
    8: "Medical Team Evaluation and Opportunities"
}

# Globals para modelo y vectorizer
_VECTORIZER = None
_NMF_MODEL = None

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

# Clasificación individual
def classify_topic(text: str) -> str:
    """Clasifica un texto corto en uno de los topics conocidos."""
    global _NMF_MODEL, _VECTORIZER
    if _NMF_MODEL is None or _VECTORIZER is None:
        raise RuntimeError("[ERROR] Call init_topic_classifier_from_db() first.")

    processed = preprocess(text)
    X = _VECTORIZER.transform([processed])
    topic_idx = int(np.argmax(_NMF_MODEL.transform(X)))
    return TOPIC_LABELS.get(topic_idx, f"Topic {topic_idx+1}")

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
            topic_name = TOPIC_LABELS[topic_idx]
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
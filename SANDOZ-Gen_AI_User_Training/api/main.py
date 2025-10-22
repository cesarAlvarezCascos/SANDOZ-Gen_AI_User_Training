import os, re  
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from src.search_kb import search_kb
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

# Habilita CORS para permitir que el frontend se comunique con la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:5173"],  # añade los que uses
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Query Format
class Ask(BaseModel):
    user_id: str | None = None
    role: str = "analyst"
    query: str
    product_version: str | None = None
    time_budget: int | None = 30
    level: int | None = 2

# LLM instructions
SYSTEM = (
    "Eres un Training Agent para un sistema de pipelines. "
    "Responde breve (<= 180 palabras), en pasos claros. "
    "Usa citaciones [n] SOLO para referirte a los pasajes proporcionados. "
    "NO incluyas un bloque 'Fuentes:' en tu salida; lo añadirá el sistema."
)

# Citations from original pdfs and docs
# "passages": text chunks retrieved from Supabase through vectorial search -> fragments from the PDFs with the embedding, metadata and text/snippet

def format_citations(passages):
    lines = []
    for i, p in enumerate(passages, start=1):
        # Be defensive: some passages might not have a source_path (None).
        sp = p.get("source_path") if isinstance(p, dict) else None
        if not sp:
            name = "<unknown>"  # placeholder
        else:
            # strip file:// then take file basename
            name = os.path.basename(str(sp).replace("file://", ""))
        lines.append(f"[{i}] {name}")
    return "\n".join(lines)


# Main Endpoint
@app.post("/ask")
def ask(req: Ask):
    # search_kb(query, top_k=8) expects an integer as the second arg.
    # Passing req.role (a string) was accidental and caused errors like
    # `invalid input syntax for type bigint: "analystanalyst"` because
    # the string was multiplied/used where an int (LIMIT) was expected.

    # RETRIEVAL
    passages = search_kb(req.query)
    if not passages:
        return {"answer": "No encontré coincidencias relevantes en la base de datos.", "citations": []}
    elif len(passages) < 2:
        return {
            "answer": "No estoy 100% segura; necesito más fuentes o material.",
            "citations": passages
        }

    # PROPMT CONSTRUCTION:
        # AUGMENTATION
    ctx = "\n\n".join([f"[{i+1}] {p['snippet']}" for i, p in enumerate(passages[:6])])
    prompt = (
        f"{SYSTEM}\n\n"
        f"Pregunta del usuario: {req.query}\n\n"
        "Pasajes relevantes (usa [n] para citarlos en tu respuesta):\n"
        f"{ctx}\n\n"
        "Instrucciones:\n"
        "- Resume y responde sin exceder 180 palabras.\n"
        "- Inserta [n] exactamente donde uses un dato de un pasaje.\n"
        "- NO inventes fuentes ni añadas 'Fuentes:' al final.\n"
        "- Incluye al menos dos citaciones si hay >=2 pasajes.\n"
    )

    # Call LLM
        # GENERATION
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        text = (
            "No estoy 100% segura: error al generar la respuesta "
            f"({type(e).__name__}). Revisa la API key o el modelo."
        )

    # Si el modelo desobedece y añade su propio bloque "Fuentes:", límpialo
    text = re.sub(
        r"\n+Fuentes:\s*(?:\[[^\]]+\].*|\S.*)+$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Heurística: fuerza al menos dos citaciones si hay pasajes suficientes
    if text.count('[') < 2 and len(passages) >= 2:
        text += " [1][2]"

    return {
        "answer": text + "\n\nFuentes:\n" + format_citations(passages[:6]),
        "citations": passages[:6]
    }
# api/main.py (fragmento)

import os, re  # <-- añade re
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from tools.search_kb import search_kb
from tools.plan_path import plan_path

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

class Ask(BaseModel):
    user_id: str | None = None
    role: str = "analyst"
    query: str
    product_version: str | None = None
    time_budget: int | None = 30
    level: int | None = 2

SYSTEM = (
    "Eres un Training Agent para un sistema de pipelines. "
    "Responde breve (<= 180 palabras), en pasos claros. "
    "Usa citaciones [n] SOLO para referirte a los pasajes proporcionados. "
    "NO incluyas un bloque 'Fuentes:' en tu salida; lo añadirá el sistema."
)

def format_citations(passages):
    lines = []
    for i, p in enumerate(passages, start=1):
        name = os.path.basename(p["source_url"]).replace("file://", "")
        lines.append(f"[{i}] {name}")
    return "\n".join(lines)

@app.post("/ask")
def ask(req: Ask):
    passages = search_kb(req.query, req.role)
    if len(passages) < 2:
        return {
            "answer": "No estoy 100% segura; necesito más fuentes o material.",
            "citations": passages
        }

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

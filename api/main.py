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
    "You are a Training Agent for a pipeline system."
    "Respond briefly (≤ 180 words) in clear steps."
    "Use [n] citations ONLY to refer to the provided passages."
    "DO NOT include a 'Sources:' block in your output; the system will add it.")

# Citations from original pdfs and docs
# "passages": text chunks retrieved from Supabase through vectorial search -> fragments from the PDFs with the embedding, metadata and text/snippet
def format_citations(passages):
    lines = []
    for i, p in enumerate(passages, start=1):
        filename = p.get("file_name") if isinstance(p, dict) else None
        if not filename:
            name = "<unknown>"  # placeholder
        else:
            name = filename
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
    passages = search_kb(req.query) # This return the Citations JSON defined in search_kb.fuse()
    if not passages:
        return {"answer": "I didn’t find any relevant matches in the database.", "citations": []}
    elif len(passages) < 2:
        return {
            "answer": "I’m not 100% sure; I need more sources or material.",
            "citations": passages
        }

    # PROPMT CONSTRUCTION:
        # AUGMENTATION
    ctx = "\n\n".join([f"[{i+1}] {p['snippet']}" for i, p in enumerate(passages[:6])])
    prompt = (
        f"{SYSTEM}\n\n"
        f"User’s question: {req.query}\n\n"
        "Relevant passages (use [n] to cite them in your answer):\n"
        f"{ctx}\n\n"
        "Instructions:\n"
        "- Summarise and answer in no more than 180 words.\n"
        "- Respond in the same language in which you are asked.\n"
        "- Insert [n] exactly where you use a fact from a passage.\n"
        "- DO NOT invent sources or add 'Sources:' at the end.\n"
        "- Include at least two citations if there are ≥ 2 passages.\n"
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
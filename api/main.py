import os, re  
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from src.search_kb import search_kb
from src.memory import SessionMemory
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
from src.adaptive_prompting import AdaptivePromptSelector

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()
memory = SessionMemory(max_turns=3)
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
    selected_topic_name: str | None = None

class FeedbackRequest(BaseModel):
    user_id: str | None = None
    session_id: str | None = None
    query: str
    answer: str
    rating: int  # 1 = thumbs up, -1 = thumbs down
    feedback_type: str | None = "overall"
    comment: str | None = None
    citations: list | None = None
    retrieved_passages: list | None = None

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

# Selector de prompts adaptativo
adaptive_selector = AdaptivePromptSelector(max_examples=2)

# Main Endpoint
@app.post("/ask")
def ask(req: Ask):
    user_id = req.user_id or "anonymous"

    # 1. Obtener estadísticas para ajustar comportamiento
    stats = adaptive_selector.get_feedback_stats()
    
    # 2. Recuperar ejemplos similares exitosos (adaptive few-shot)
    positive_examples = adaptive_selector.retrieve_positive_examples(req.query)
    few_shot_context = adaptive_selector.build_few_shot_context(positive_examples)

    # 3. Retrieve memory history
    past_turns = memory.get(user_id)
    context_history = "\n".join(
        [f"Q: {t['query']}\nA: {t['answer']}" for t in past_turns]
    )

    # 4. Search knowledge base
    search_result = search_kb(req.query, selected_topic_name=req.selected_topic_name)
    if isinstance(search_result, dict) and search_result.get("status") == "choose_topic":
        return search_result # user selects topic

    # 5. Retrieve relevant knowledge base snippets
    passages = search_result
    if not passages:
        return {"answer": "I didn’t find any relevant matches in the database.", "citations": []}

    # 6. Build the augmented prompt + few-shot examples
    ctx = "\n\n".join([f"[{i+1}] {p['snippet']}" for i, p in enumerate(passages[:6])])

    # Ajustar instrucciones según satisfaction rate
    tone_adjustment = ""
    print(f"Satisfaction rate: {stats['satisfaction_rate']}")
    if stats['satisfaction_rate'] < 0.65 and stats['total'] > 10:
        tone_adjustment = "\n- IMPORTANT: Users have reported unsatifaction. Be more clear and specific."
    
    prompt = (
        f"{SYSTEM}\n\n"
        f"{few_shot_context}\n\n" # Few-shot examples
        f"Conversation history:\n{context_history or '(none)'}\n\n"
        f"User’s question: {req.query}\n\n"
        f"Relevant passages (use [n] to cite them in your answer):\n"
        f"{ctx}\n\n"
        "Instructions:\n"
        "- Summarise and answer in no more than 180 words.\n"
        "- Respond in the same language in which you are asked.\n"
        "- Insert [n] exactly where you use a fact from a passage.\n"
        "- DO NOT invent sources or add 'Sources:' at the end.\n"
        "- Include at least two citations if there are ≥ 2 passages.\n"
        f"{tone_adjustment}" # si la satisfacción es baja, ajustar tono (más claro y específico)
    )

    # Generate with OpenAI
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        text = f"Error generating response: {type(e).__name__}."

    # Clean and finalize
    text = re.sub(r"\n+Fuentes:\s*(?:\[[^\]]+\].*|\S.*)+$", "", text, flags=re.IGNORECASE | re.DOTALL)
    if text.count('[') < 2 and len(passages) >= 2:
        text += " [1][2]"

    final_answer = text + "\n\nFuentes:\n" + format_citations(passages[:6])

    # Save to memory
    memory.add(user_id, req.query, text)

    return {"answer": final_answer, "citations": passages[:6],
        "adaptive_stats": stats # para debug (opcional)
        }

class FeedbackRequest(BaseModel):
    user_id: str | None = None
    session_id: str | None = None
    query: str
    answer: str
    rating: int  # 1 = thumbs up, -1 = thumbs down
    feedback_type: str | None = "overall"
    comment: str | None = None
    citations: list | None = None
    retrieved_passages: list | None = None

# Añade este endpoint después de /ask
@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """Captura feedback del usuario para mejorar el sistema"""
    try:
        from supabase import create_client
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

        # Generar embedding de la consulta
        query_embedding = None
        try:
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=req.query
            )
            query_embedding = embedding_response.data[0].embedding
        except Exception as e:
            print(f"Warning: Could not generate embedding: {e}")

        query_embedding = embedding_response.data[0].embedding
        data = {
            "user_id": req.user_id or "anonymous",
            "session_id": req.session_id,
            "query": req.query,
            "answer": req.answer,
            "rating": req.rating,
            "feedback_type": req.feedback_type,
            "comment": req.comment,
            "citations": req.citations,
            "retrieved_passages": req.retrieved_passages,
            "model_used": "gpt-4o-mini",
            "query_embedding": query_embedding # NUEVO: guardar embedding de la consulta
        }
        
        result = supabase.table("feedback").insert(data).execute()
        
        if not result.data:
            return {
                "status": "warning",
                "message": "Feedback no guardado - verifica RLS policies"
            }
        
        return {
            "status": "success",
            "message": "Feedback guardado correctamente",
            "feedback_id": result.data[0]["id"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error al guardar feedback: {str(e)}"
        }


@app.get("/feedback/stats")
def get_feedback_stats():
    """Estadísticas incluyendo cobertura de embeddings"""
    try:
        from supabase import create_client
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
        positive = supabase.table("feedback").select("*").eq("rating", 1).execute()
        negative = supabase.table("feedback").select("*").eq("rating", -1).execute()
        
        # Contar embeddings
        with_embeddings = supabase.table("feedback")\
            .select("id")\
            .not_.is_("query_embedding", "null")\
            .execute()
        
        total = len(positive.data) + len(negative.data)
        
        return {
            "total_feedback": total,
            "positive": len(positive.data),
            "negative": len(negative.data),
            "satisfaction_rate": len(positive.data) / total * 100 if total > 0 else 0,
            "embeddings_coverage": len(with_embeddings.data) / total * 100 if total > 0 else 0,
            "embeddings_count": len(with_embeddings.data)
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/adaptive/status")
def adaptive_status():
    """Muestra estado del sistema adaptativo"""
    try:
        from supabase import create_client
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        stats = adaptive_selector.get_feedback_stats()
        
        # Obtener ejemplos más usados
        recent_positive = supabase.table("feedback")\
            .select("query")\
            .eq("rating", 1)\
            .order("created_at", desc=True)\
            .limit(5)\
            .execute()
        
        return {
            "feedback_stats": stats,
            "recent_positive_queries": [r['query'] for r in recent_positive.data],
            "adaptive_enabled": True,
            "max_examples": adaptive_selector.max_examples
        }
    except Exception as e:
        return {"error": str(e)}
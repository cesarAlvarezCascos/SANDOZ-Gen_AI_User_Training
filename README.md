# Requisitos
- Python 3.12+
- PostgreSQL 15+ con extensión [pgvector](https://github.com/pgvector/pgvector)
- [Docker](https://www.docker.com/) (opcional, para levantar la base de datos)
- OpenAI API Key

# Instalación
1. Crear carpeta (trainer-workspace)
2. Clonar repositorio (https://github.com/openai/openai-agents-python.git)
3. Clonar este repositorio 
4. Crear entrono virtual e instalar dependencias (pip install -r requirements.txt)
5. Crear carpeta (pdfs/) y meter dentro los 4 pdfs que ha enviado david
5. Crear la base de datos 
```bash
docker compose up -d db
psql "postgresql://postgres:postgres@localhost:5432/trainer" -f schema/init.sql
```
6. Crear un .env como el de ejemplo sustituyendo la api
7. Ejecutar (python ingest/ingest_pdfs.py)
8. Arrancar la API (uvicorn api.main:app --reload --port 8000). Una vez arrancada está aqui (http://127.0.0.1:8000/docs)
9. Ejemplo (curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"role":"analyst","query":"¿Qué es LoE y por qué puede cambiar?"}')
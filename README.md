# Requisitos
- Python 3.12+
- PostgreSQL 15+ con extensión [pgvector](https://github.com/pgvector/pgvector)
- [Docker](https://www.docker.com/) (opcional, para levantar la base de datos)
- OpenAI API Key

# Instalación
1. Clonar este repositorio 
2. Crear entrono virtual e instalar dependencias (pip install -r requirements.txt)
3. Crear carpeta (pdfs/) y meter dentro los 4 pdfs que ha enviado david
4. Crear la base de datos 
```bash
docker compose up -d db
psql "postgresql://postgres:postgres@localhost:5432/trainer" -f schema/init.sql
```
5. Crear un .env como el de ejemplo sustituyendo la api
6. Ejecutar (python ingest/ingest_pdfs.py)
7. Arrancar la API (uvicorn api.main:app --reload --port 8000). Una vez arrancada está aqui (http://127.0.0.1:8000/docs)
8. Ejemplo (curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"role":"analyst","query":"¿Qué es LoE y por qué puede cambiar?"}')
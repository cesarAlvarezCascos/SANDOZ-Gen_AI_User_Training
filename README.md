# Instalar Dependencias y modelo de lenguaje de spaCy
```bash
pip install -r requirements.txt
```

```bash
python -m spacy download en_core_web_lg
```

# Runear el agente
1. Arrancar la api
```bash
uvicorn api.main:app --reload --port 8000
```
2. Ejecutar
```bash
python -m http.server 5500
```
Una vez ejecutado, el agente está aqui: http://localhost:5500/chat.html
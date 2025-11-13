# src/import_topics_csv.py
import os
import csv
import psycopg
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
conn = psycopg.connect(os.getenv("DATABASE_URL"))

def _parse_bool(val: str | None):
    if val is None:
        return None
    v = val.strip().lower()
    if v in ("true", "t", "1", "yes", "y", "sí", "si"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    return None  # si está vacío o raro, no tocamos is_active

def _parse_keywords_cell(val: str | None):
    """
    Convierte 'kw1, kw2, kw3' -> ['kw1','kw2','kw3']
    Si está vacío, devuelve None (para guardar NULL).
    """
    if not val:
        return None
    parts = [p.strip() for p in val.split(",")]
    kws = [p for p in parts if p]
    return kws or None

def import_topics_csv(filename: str = "topics_export.csv"):
    """
    Importa cambios desde el CSV de Descargas a la tabla topics_test.
    - Actualiza solo campos MANUALES + is_active en filas existentes.
    - Inserta filas nuevas cuando no hay id pero sí contenido manual.
    """
    downloads_dir = Path.home() / "Downloads"
    in_path = downloads_dir / filename

    if not in_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {in_path}")

    print(f"[INFO] Importing topics from {in_path.resolve()}")

    updated = 0
    inserted = 0
    skipped = 0

    with in_path.open("r", encoding="utf-8") as f, conn, conn.cursor() as cur:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')

        for row in reader:
            row_id = (row.get("id") or "").strip()
            model_idx_raw = (row.get("model_topic_index") or "").strip()
            name_manual_raw = (row.get("name_manual") or "").strip()
            kw_manual_raw = (row.get("key_words_manual") or "").strip()
            is_active_raw = (row.get("is_active") or "").strip()
            descripcion_raw = (row.get("descripcion") or "").strip()
            descripcion_manual_raw = (row.get("descripcion_manual") or "").strip()

            # Parse
            name_manual = name_manual_raw or None
            key_words_manual = _parse_keywords_cell(kw_manual_raw)
            is_active = _parse_bool(is_active_raw)
            descripcion = descripcion_raw or None
            descripcion_manual = descripcion_manual_raw or None

            # Caso 1: fila ya existente (id presente) -> UPDATE
            if row_id:
                # Si no han tocado nada manual ni is_active, nos lo saltamos
                if (
                    name_manual is None
                    and key_words_manual is None
                    and is_active is None
                    and descripcion_manual is None
                ):
                    skipped += 1
                    continue

                # UPDATE: solo campos manuales + is_active 
                cur.execute(
                    """
                    UPDATE topics_test
                    SET
                        name_manual = %s,
                        key_words_manual = %s,
                        is_active = COALESCE(%s, is_active),
                        descripcion_manual = %s
                    WHERE id = %s;
                    """,
                    (name_manual, key_words_manual, is_active, descripcion_manual, row_id)
                )
                updated += 1
                continue

            # Caso 2: no hay id -> potencial topic nuevo
            # Si no hay nada manual ni is_active ni descripción manual, ignoramos
            if not (name_manual or key_words_manual or is_active is not None or descripcion_manual):
                skipped += 1
                continue

            # Parse model_topic_index si viene
            model_idx = None
            if model_idx_raw:
                try:
                    model_idx = int(model_idx_raw)
                except ValueError:
                    model_idx = None

            # Nombre, keywords y descripción "efectivos" para rellenar campos auto
            effective_name = name_manual or "Manual topic"
            effective_kws = key_words_manual or []
            effective_descripcion = descripcion_manual or descripcion

            cur.execute(
                """
                INSERT INTO topics_test
                    (model_topic_index, name, key_words,
                     name_manual, key_words_manual, is_active,
                     descripcion, descripcion_manual)
                VALUES (%s, %s, %s, %s, %s, COALESCE(%s, TRUE), %s, %s);
                """,
                (
                    model_idx,
                    effective_name,
                    effective_kws,
                    name_manual,
                    key_words_manual,
                    is_active,
                    effective_descripcion,
                    descripcion_manual
                )
            )
            inserted += 1

    print(f"[INFO] Import finished. Updated: {updated}, Inserted: {inserted}, Skipped: {skipped}")

if __name__ == "__main__":
    import_topics_csv()

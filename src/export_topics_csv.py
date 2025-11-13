# src/export_topics_test.py
import os
import csv
import psycopg
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
conn = psycopg.connect(os.getenv("DATABASE_URL"))

def export_topics_csv(filename: str = "topics_export.csv"):
    """
    Exporta la tabla topics_test a un CSV listo para abrir en Excel.
    - Usa ';' como delimitador (típico en Excel en ES).
    - Guarda el archivo en la carpeta Descargas del usuario.
    """
    query = """
        SELECT
            id,
            model_topic_index,
            name        AS name_auto,
            key_words   AS key_words_auto,
            name_manual,
            key_words_manual,
            is_active,
            descripcion,
            descripcion_manual
        FROM topics_test
        ORDER BY model_topic_index NULLS LAST, created_at;
    """

    with conn, conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]

    # Carpeta Descargas del usuario
    downloads_dir = Path.home() / "Downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)  # por si acaso
    out_path = downloads_dir / filename

    print(f"[INFO] Exporting {len(rows)} topics to {out_path.resolve()}")

    # Abrimos el CSV con delimitador ';' para que Excel lo lea bien
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(
            f,
            delimiter=';',          # separador de campos
            quotechar='"',          # para encerrar textos con ; o comas
            quoting=csv.QUOTE_MINIMAL
        )
        # Cabecera
        writer.writerow(colnames)

        for row in rows:
            row_list = list(row)

            # key_words_auto = col 3, key_words_manual = col 5 (0-based: 3 y 5)
            for idx in (3, 5):
                if idx < len(row_list):
                    val = row_list[idx]
                    if isinstance(val, list):
                        # Convertimos el array en "kw1, kw2, kw3" dentro de la celda
                        row_list[idx] = ", ".join(str(x) for x in val)
                    elif val is None:
                        row_list[idx] = ""

            writer.writerow(row_list)

    print("[INFO] Export complete.")

if __name__ == "__main__":
    export_topics_csv()

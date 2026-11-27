"""Carga el CSV del data lake a DuckDB.  Uso:  python -m src.ingest"""
import duckdb

from src.config import DATA_RAW, CATALOGO, DUCKDB_PATH, TABLA


def conectar(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def ingest():
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = conectar(read_only=False)
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLA} AS
        SELECT CAST(fecha AS DATE) AS fecha, pais, site, funnel, creative,
               CAST(media_cost AS DOUBLE) AS media_cost,
               impressions, clicks, subscriptions, lpv, video_starts, video_completes
        FROM read_csv_auto('{DATA_RAW}', header=true)
    """)
    con.execute(f"CREATE OR REPLACE TABLE catalogo_creativos AS SELECT * FROM read_csv_auto('{CATALOGO}')")
    n = con.execute(f"SELECT COUNT(*) FROM {TABLA}").fetchone()[0]
    con.close()
    print(f"OK: {n:,} filas en {DUCKDB_PATH.name}:{TABLA}")


if __name__ == "__main__":
    ingest()

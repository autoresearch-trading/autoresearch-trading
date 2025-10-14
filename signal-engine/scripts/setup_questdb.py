#!/usr/bin/env python3
"""Initialize QuestDB schema for the signal engine."""

from pathlib import Path

import psycopg


def main() -> None:
    schema_path = Path(__file__).resolve().parent.parent / "src" / "db" / "schema.sql"
    sql = schema_path.read_text()

    conn_string = "host=localhost port=8812 user=admin password=quest dbname=qdb"
    with psycopg.connect(conn_string) as conn:
        conn.execute(sql)
        conn.commit()

    print("✓ QuestDB tables created")


if __name__ == "__main__":
    main()

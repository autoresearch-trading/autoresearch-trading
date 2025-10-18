#!/usr/bin/env python3
"""
Initialize QuestDB schema for local development (non-Docker).

This wrapper adds friendlier guidance around `setup_questdb.py` for engineers
who want to target a locally running QuestDB instance.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import psycopg  # type: ignore
except ImportError:
    print("Error: psycopg package not found. Install requirements first:")
    print("  cd signal-engine && pip install -r requirements.txt")
    sys.exit(1)


def main() -> None:
    schema_path = Path(__file__).resolve().parent.parent / "src" / "db" / "schema.sql"

    if not schema_path.exists():
        print(f"Error: Schema file not found at {schema_path}")
        sys.exit(1)

    sql = schema_path.read_text()

    conn_string = "host=localhost port=8812 user=admin password=quest dbname=qdb"

    print("Connecting to local QuestDB at localhost:8812...")

    try:
        with psycopg.connect(conn_string, connect_timeout=5) as conn:
            print("Connected successfully. Creating tables...")
            conn.execute(sql)
            conn.commit()
            print("✓ QuestDB tables created successfully!")
            print("\nYou can access QuestDB at:")
            print("  HTTP Console: http://localhost:9000")
            print("  PostgreSQL:   localhost:8812")
    except psycopg.OperationalError as exc:
        print(f"\n✗ Failed to connect to QuestDB: {exc}")
        print("\nMake sure QuestDB is running:")
        print("  1. Download from: https://questdb.io/download/")
        print("  2. Or use Docker: make questdb-docker")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"\n✗ Error creating tables: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv(Path(__file__).parent / ".env")

_db_engine: Engine | None = None


def resolve_db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL no configurada. Define DATABASE_URL en back/config/.env")
    return url


def get_db_engine() -> Engine:
    global _db_engine

    if _db_engine is None:
        _db_engine = create_engine(resolve_db_url(), pool_pre_ping=True, future=True)

    return _db_engine


def init_database() -> None:
    engine = get_db_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS saved_detections (
                    id SERIAL PRIMARY KEY,
                    nombre VARCHAR(200) NOT NULL,
                    imagen TEXT NOT NULL,
                    descripcion TEXT
                )
                """
            )
        )

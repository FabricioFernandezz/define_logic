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
        # --- Multi-tenant: industrias y usuarios ---
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS industries (
                    id SERIAL PRIMARY KEY,
                    nombre VARCHAR(200) NOT NULL,
                    owner_user_id INTEGER,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    nombre VARCHAR(200) NOT NULL,
                    rol VARCHAR(20) NOT NULL DEFAULT 'member',
                    industry_id INTEGER NOT NULL REFERENCES industries(id),
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        # Whitelist de mails autorizados por industria (invitaciones)
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS industry_allowed_emails (
                    id SERIAL PRIMARY KEY,
                    industry_id INTEGER NOT NULL REFERENCES industries(id),
                    email VARCHAR(255) NOT NULL UNIQUE,
                    used BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
        )

        # --- Datos de la app, scoped por industria ---
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
        # epp_zone_config.id == industry_id (una config por industria)
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS epp_zone_config (
                    id INTEGER PRIMARY KEY,
                    zones TEXT NOT NULL DEFAULT '[]',
                    default_zone_epp TEXT NOT NULL DEFAULT '[]',
                    default_zone_active BOOLEAN NOT NULL DEFAULT TRUE,
                    default_zone_require_person BOOLEAN NOT NULL DEFAULT FALSE
                )
                """
            )
        )

        # --- Migracion: agregar industry_id a tablas existentes ---
        connection.execute(
            text(
                "ALTER TABLE saved_detections ADD COLUMN IF NOT EXISTS industry_id INTEGER"
            )
        )

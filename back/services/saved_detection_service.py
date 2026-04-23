from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from back.config.database import get_db_engine, init_database
from back.models.saved_detection import SavedDetectionCreate


class DuplicateSavedDetectionError(RuntimeError):
    def __init__(self, existing: Dict[str, Any]):
        super().__init__("Imagen ya fue guardada previamente")
        self.existing = existing


def list_saved_detections() -> List[Dict[str, Any]]:
    init_database()

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            rows = (
                connection.execute(
                    text(
                        """
                        SELECT id, nombre, imagen, descripcion
                        FROM saved_detections
                        ORDER BY id DESC
                        """
                    )
                )
                .mappings()
                .all()
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo consultar detecciones guardadas: {exc}") from exc

    return [
        {
            "id": row["id"],
            "nombre": row["nombre"],
            "imagen": row["imagen"],
            "descripcion": row["descripcion"],
        }
        for row in rows
    ]


def create_saved_detection(payload: SavedDetectionCreate) -> Dict[str, Any]:
    init_database()

    nombre = payload.nombre.strip()
    if not nombre:
        raise ValueError("El nombre es obligatorio")

    descripcion = payload.descripcion.strip() if payload.descripcion else None

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            existing = (
                connection.execute(
                    text(
                        """
                        SELECT id, nombre, imagen, descripcion
                        FROM saved_detections
                        WHERE imagen = :imagen
                        ORDER BY id DESC
                        LIMIT 1
                        """
                    ),
                    {
                        "imagen": payload.imagen,
                    },
                )
                .mappings()
                .first()
            )
            if existing is not None:
                raise DuplicateSavedDetectionError(
                    {
                        "id": existing["id"],
                        "nombre": existing["nombre"],
                        "imagen": existing["imagen"],
                        "descripcion": existing["descripcion"],
                    }
                )

            row = (
                connection.execute(
                    text(
                        """
                        INSERT INTO saved_detections (nombre, imagen, descripcion)
                        VALUES (:nombre, :imagen, :descripcion)
                        RETURNING id, nombre, imagen, descripcion
                        """
                    ),
                    {
                        "nombre": nombre,
                        "imagen": payload.imagen,
                        "descripcion": descripcion,
                    },
                )
                .mappings()
                .one()
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo guardar detección: {exc}") from exc

    return {
        "id": row["id"],
        "nombre": row["nombre"],
        "imagen": row["imagen"],
        "descripcion": row["descripcion"],
    }

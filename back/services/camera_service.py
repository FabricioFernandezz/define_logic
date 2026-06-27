from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from back.config.database import get_db_engine, init_database
from back.models.camera import CameraCreate, CameraUpdate


class CameraNotFoundError(RuntimeError):
    pass


def _row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "nombre": row["nombre"],
        "url": row["url"],
        "activa": bool(row["activa"]),
    }


def list_cameras(industry_id: int) -> List[Dict[str, Any]]:
    init_database()

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            rows = (
                connection.execute(
                    text(
                        """
                        SELECT id, nombre, url, activa
                        FROM cameras
                        WHERE industry_id = :industry_id
                        ORDER BY id ASC
                        """
                    ),
                    {"industry_id": industry_id},
                )
                .mappings()
                .all()
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo consultar cámaras: {exc}") from exc

    return [_row_to_dict(row) for row in rows]


def create_camera(payload: CameraCreate, industry_id: int) -> Dict[str, Any]:
    init_database()

    nombre = payload.nombre.strip()
    url = payload.url.strip()
    if not nombre:
        raise ValueError("El nombre es obligatorio")
    if not url:
        raise ValueError("La URL es obligatoria")

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        INSERT INTO cameras (industry_id, nombre, url, activa)
                        VALUES (:industry_id, :nombre, :url, :activa)
                        RETURNING id, nombre, url, activa
                        """
                    ),
                    {
                        "industry_id": industry_id,
                        "nombre": nombre,
                        "url": url,
                        "activa": payload.activa,
                    },
                )
                .mappings()
                .one()
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo crear cámara: {exc}") from exc

    return _row_to_dict(row)


def update_camera(camera_id: int, payload: CameraUpdate, industry_id: int) -> Dict[str, Any]:
    init_database()

    fields: Dict[str, Any] = {}
    if payload.nombre is not None:
        nombre = payload.nombre.strip()
        if not nombre:
            raise ValueError("El nombre no puede estar vacío")
        fields["nombre"] = nombre
    if payload.url is not None:
        url = payload.url.strip()
        if not url:
            raise ValueError("La URL no puede estar vacía")
        fields["url"] = url
    if payload.activa is not None:
        fields["activa"] = payload.activa

    if not fields:
        raise ValueError("No hay cambios para aplicar")

    set_clause = ", ".join(f"{col} = :{col}" for col in fields)
    params = {**fields, "id": camera_id, "industry_id": industry_id}

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        f"""
                        UPDATE cameras
                        SET {set_clause}
                        WHERE id = :id AND industry_id = :industry_id
                        RETURNING id, nombre, url, activa
                        """
                    ),
                    params,
                )
                .mappings()
                .first()
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo actualizar cámara: {exc}") from exc

    if row is None:
        raise CameraNotFoundError("Cámara no encontrada")

    return _row_to_dict(row)


def delete_camera(camera_id: int, industry_id: int) -> None:
    init_database()

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    DELETE FROM cameras
                    WHERE id = :id AND industry_id = :industry_id
                    """
                ),
                {"id": camera_id, "industry_id": industry_id},
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo eliminar cámara: {exc}") from exc

    if result.rowcount == 0:
        raise CameraNotFoundError("Cámara no encontrada")

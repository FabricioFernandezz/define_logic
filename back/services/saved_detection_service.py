from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from back.config.database import get_db_engine, init_database
from back.models.saved_detection import SavedDetectionCreate

# Evidence images are stored on the filesystem under storage/evidences/YYYY/MM/DD/<sha256>.<ext>
# and the DB keeps only the public /media-relative path. This avoids bloating PostgreSQL with
# base64 blobs (~33% larger than the raw bytes) and lets the images be served by StaticFiles.
STORAGE_ROOT = Path(__file__).resolve().parents[1] / "storage"
_EVIDENCE_SUBDIR = "evidences"
_MEDIA_PREFIX = "/media/"
_DATA_URL_RE = re.compile(r"^data:image/(?P<ext>[a-zA-Z0-9.+-]+);base64,(?P<data>.+)$", re.DOTALL)
_EXT_ALIASES = {"jpeg": "jpg"}
_ALLOWED_EXT = {"jpg", "png", "webp", "gif", "bmp"}


def ensure_storage_dirs() -> None:
    (STORAGE_ROOT / _EVIDENCE_SUBDIR).mkdir(parents=True, exist_ok=True)


def _persist_image(imagen: str) -> str:
    """Decode a base64 data URL, write it to evidences/YYYY/MM/DD/<sha256>.<ext> and
    return its public /media path. Non data-URL inputs (already a path, or legacy value)
    are returned untouched for backward compatibility."""
    match = _DATA_URL_RE.match((imagen or "").strip())
    if not match:
        return imagen  # already a path / not a data URL → store verbatim

    ext = match.group("ext").lower()
    ext = _EXT_ALIASES.get(ext, ext)
    if ext not in _ALLOWED_EXT:
        ext = "jpg"

    try:
        raw = base64.b64decode(match.group("data"), validate=True)
    except (ValueError, TypeError):
        return imagen  # malformed base64 → keep verbatim rather than lose the data

    digest = hashlib.sha256(raw).hexdigest()
    now = datetime.now(timezone.utc)
    rel_dir = Path(_EVIDENCE_SUBDIR) / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"
    abs_dir = STORAGE_ROOT / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{digest}.{ext}"
    abs_path = abs_dir / filename
    if not abs_path.exists():  # same content → same hash → write once
        abs_path.write_bytes(raw)

    return _MEDIA_PREFIX + (rel_dir / filename).as_posix()


def resolve_evidence_file(imagen_media_path: str, industry_id: int) -> Path | None:
    """Map a stored /media path to its on-disk file, but only if a saved_detections row
    with that path belongs to `industry_id`. Returns None when the caller's industry does
    not own the image (tenant isolation), the path escapes STORAGE_ROOT, or the file is gone."""
    if not imagen_media_path.startswith(_MEDIA_PREFIX):
        return None

    init_database()
    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            owned = connection.execute(
                text(
                    "SELECT 1 FROM saved_detections WHERE imagen = :img AND industry_id = :iid LIMIT 1"
                ),
                {"img": imagen_media_path, "iid": industry_id},
            ).first()
    except SQLAlchemyError:
        return None
    if owned is None:
        return None

    rel = imagen_media_path[len(_MEDIA_PREFIX):]
    try:
        abs_path = (STORAGE_ROOT / rel).resolve()
        abs_path.relative_to(STORAGE_ROOT.resolve())  # raises if outside storage
    except (ValueError, OSError):
        return None
    return abs_path if abs_path.is_file() else None


def _remove_evidence_file(imagen: str | None) -> None:
    """Delete the backing file for a /media path. Legacy base64 rows have no file → skip.
    Guards against path traversal: only unlinks files inside STORAGE_ROOT."""
    if not imagen or not imagen.startswith(_MEDIA_PREFIX):
        return
    rel = imagen[len(_MEDIA_PREFIX):]
    try:
        abs_path = (STORAGE_ROOT / rel).resolve()
        abs_path.relative_to(STORAGE_ROOT.resolve())  # raises if outside storage
    except (ValueError, OSError):
        return
    try:
        abs_path.unlink(missing_ok=True)
    except OSError:
        pass


class DuplicateSavedDetectionError(RuntimeError):
    def __init__(self, existing: Dict[str, Any]):
        super().__init__("Imagen ya fue guardada previamente")
        self.existing = existing


class SavedDetectionNotFoundError(RuntimeError):
    pass


def list_saved_detections(industry_id: int) -> List[Dict[str, Any]]:
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
                        WHERE industry_id = :industry_id
                        ORDER BY id DESC
                        """
                    ),
                    {"industry_id": industry_id},
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


def create_saved_detection(payload: SavedDetectionCreate, industry_id: int) -> Dict[str, Any]:
    init_database()

    nombre = payload.nombre.strip()
    if not nombre:
        raise ValueError("El nombre es obligatorio")

    descripcion = payload.descripcion.strip() if payload.descripcion else None

    # Write the image to disk once; the DB stores only the /media path. Deterministic by
    # content hash, so re-saving the same image reuses the same file and the dedup below
    # catches it by path.
    stored_imagen = _persist_image(payload.imagen)

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            existing = (
                connection.execute(
                    text(
                        """
                        SELECT id, nombre, imagen, descripcion
                        FROM saved_detections
                        WHERE imagen = :imagen AND industry_id = :industry_id
                        ORDER BY id DESC
                        LIMIT 1
                        """
                    ),
                    {
                        "imagen": stored_imagen,
                        "industry_id": industry_id,
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
                        INSERT INTO saved_detections (nombre, imagen, descripcion, industry_id)
                        VALUES (:nombre, :imagen, :descripcion, :industry_id)
                        RETURNING id, nombre, imagen, descripcion
                        """
                    ),
                    {
                        "nombre": nombre,
                        "imagen": stored_imagen,
                        "descripcion": descripcion,
                        "industry_id": industry_id,
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


def delete_saved_detection(detection_id: int, industry_id: int) -> None:
    init_database()

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            deleted = (
                connection.execute(
                    text(
                        """
                        DELETE FROM saved_detections
                        WHERE id = :id AND industry_id = :industry_id
                        RETURNING imagen
                        """
                    ),
                    {"id": detection_id, "industry_id": industry_id},
                )
                .mappings()
                .first()
            )

            still_referenced = False
            if deleted is not None and deleted["imagen"]:
                still_referenced = (
                    connection.execute(
                        text(
                            "SELECT 1 FROM saved_detections WHERE imagen = :imagen LIMIT 1"
                        ),
                        {"imagen": deleted["imagen"]},
                    ).first()
                    is not None
                )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo eliminar detección: {exc}") from exc

    if deleted is None:
        raise SavedDetectionNotFoundError("Registro no encontrado")

    # Same content can be shared across tenants (same hash, same day → same file).
    # Only remove the backing file when no row references it anymore.
    if not still_referenced:
        _remove_evidence_file(deleted["imagen"])

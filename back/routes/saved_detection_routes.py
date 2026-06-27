from __future__ import annotations

from typing import Any, Dict, Optional

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse

from back.controllers.saved_detection_controller import (
    create_saved_detection_controller,
    delete_saved_detection_controller,
    list_saved_detections_controller,
)
from back.dependencies import get_current_user
from back.models.saved_detection import SavedDetectionCreate
from back.services.auth_service import decode_access_token
from back.services.saved_detection_service import resolve_evidence_file

router = APIRouter()


@router.get("/api/saved-detections")
def list_saved_detections(current_user: Dict[str, Any] = Depends(get_current_user)):
    return list_saved_detections_controller(current_user["industry_id"])


@router.post("/api/saved-detections")
def create_saved_detection(
    payload: SavedDetectionCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return create_saved_detection_controller(payload, current_user["industry_id"])


@router.delete("/api/saved-detections/{detection_id}")
def delete_saved_detection(
    detection_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return delete_saved_detection_controller(detection_id, current_user["industry_id"])


@router.get("/api/saved-detections/image/{file_path:path}")
def get_evidence_image(
    file_path: str,
    token: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    """Serve an evidence image only to an authenticated user whose industry owns it.
    Token comes from the `token` query param (an <img> tag can't set headers) and falls
    back to the Authorization header for programmatic callers."""
    raw = token
    if not raw and authorization and authorization.lower().startswith("bearer "):
        raw = authorization.split(" ", 1)[1].strip()
    if not raw:
        raise HTTPException(status_code=401, detail="No autenticado")

    try:
        payload = decode_access_token(raw)
        industry_id = int(payload["industry_id"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Token inválido")

    abs_path = resolve_evidence_file("/media/" + file_path, industry_id)
    if abs_path is None:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    return FileResponse(str(abs_path))

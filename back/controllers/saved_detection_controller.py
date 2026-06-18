from __future__ import annotations

from typing import Any, Dict, List

from fastapi import HTTPException

from back.models.saved_detection import SavedDetectionCreate
from back.services.saved_detection_service import (
    DuplicateSavedDetectionError,
    SavedDetectionNotFoundError,
    create_saved_detection,
    delete_saved_detection,
    list_saved_detections,
)


def list_saved_detections_controller(industry_id: int) -> List[Dict[str, Any]]:
    try:
        return list_saved_detections(industry_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def create_saved_detection_controller(payload: SavedDetectionCreate, industry_id: int) -> Dict[str, Any]:
    try:
        return create_saved_detection(payload, industry_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DuplicateSavedDetectionError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "existing": exc.existing,
            },
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def delete_saved_detection_controller(detection_id: int, industry_id: int) -> Dict[str, Any]:
    try:
        delete_saved_detection(detection_id, industry_id)
        return {"ok": True, "id": detection_id}
    except SavedDetectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import HTTPException

from back.models.saved_detection import SavedDetectionCreate
from back.services.saved_detection_service import (
    DuplicateSavedDetectionError,
    create_saved_detection,
    list_saved_detections,
)


def list_saved_detections_controller() -> List[Dict[str, Any]]:
    try:
        return list_saved_detections()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def create_saved_detection_controller(payload: SavedDetectionCreate) -> Dict[str, Any]:
    try:
        return create_saved_detection(payload)
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

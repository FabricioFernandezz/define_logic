from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from back.controllers.saved_detection_controller import (
    create_saved_detection_controller,
    delete_saved_detection_controller,
    list_saved_detections_controller,
)
from back.dependencies import get_current_user
from back.models.saved_detection import SavedDetectionCreate

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

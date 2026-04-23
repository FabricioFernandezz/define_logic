from __future__ import annotations

from fastapi import APIRouter

from back.controllers.saved_detection_controller import (
    create_saved_detection_controller,
    list_saved_detections_controller,
)
from back.models.saved_detection import SavedDetectionCreate

router = APIRouter()


@router.get("/api/saved-detections")
def list_saved_detections():
    return list_saved_detections_controller()


@router.post("/api/saved-detections")
def create_saved_detection(payload: SavedDetectionCreate):
    return create_saved_detection_controller(payload)

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from back.controllers.camera_controller import (
    create_camera_controller,
    delete_camera_controller,
    list_cameras_controller,
    update_camera_controller,
)
from back.dependencies import get_current_user
from back.models.camera import CameraCreate, CameraUpdate

router = APIRouter()


@router.get("/api/cameras")
def list_cameras_route(current_user: Dict[str, Any] = Depends(get_current_user)):
    return list_cameras_controller(current_user["industry_id"])


@router.post("/api/cameras")
def create_camera_route(
    payload: CameraCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return create_camera_controller(payload, current_user["industry_id"])


@router.put("/api/cameras/{camera_id}")
def update_camera_route(
    camera_id: int,
    payload: CameraUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return update_camera_controller(camera_id, payload, current_user["industry_id"])


@router.delete("/api/cameras/{camera_id}")
def delete_camera_route(
    camera_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return delete_camera_controller(camera_id, current_user["industry_id"])

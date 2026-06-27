from __future__ import annotations

from typing import Any, Dict, List

from fastapi import HTTPException

from back.models.camera import CameraCreate, CameraUpdate
from back.services.camera_service import (
    CameraNotFoundError,
    create_camera,
    delete_camera,
    list_cameras,
    update_camera,
)


def list_cameras_controller(industry_id: int) -> List[Dict[str, Any]]:
    try:
        return list_cameras(industry_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def create_camera_controller(payload: CameraCreate, industry_id: int) -> Dict[str, Any]:
    try:
        return create_camera(payload, industry_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def update_camera_controller(camera_id: int, payload: CameraUpdate, industry_id: int) -> Dict[str, Any]:
    try:
        return update_camera(camera_id, payload, industry_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def delete_camera_controller(camera_id: int, industry_id: int) -> Dict[str, Any]:
    try:
        delete_camera(camera_id, industry_id)
        return {"ok": True, "id": camera_id}
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

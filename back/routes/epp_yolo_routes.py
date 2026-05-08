from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from back.controllers.epp_yolo_controller import (
    detect_epp_frame_controller,
    detect_epp_image_controller,
    get_epp_classes_controller,
)

router = APIRouter()


@router.get("/api/epp/classes")
async def epp_get_classes():
    return await get_epp_classes_controller()


@router.post("/api/epp/detect-image")
async def epp_detect_image(file: UploadFile = File(...)):
    return await detect_epp_image_controller(file)


@router.post("/api/epp/detect-frame")
async def epp_detect_frame(
    file: UploadFile = File(...),
    zones: str = Form(default=None),
    default_zone_epp: str = Form(default=None),
    default_zone_active: str = Form(default="true"),
    default_zone_require_person: str = Form(default="false"),
):
    return await detect_epp_frame_controller(file, zones, default_zone_epp, default_zone_active, default_zone_require_person)

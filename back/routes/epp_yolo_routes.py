from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from back.controllers.epp_yolo_controller import (
    detect_epp_frame_controller,
    detect_epp_image_controller,
)

router = APIRouter()


@router.post("/api/epp/detect-image")
async def epp_detect_image(file: UploadFile = File(...)):
    return await detect_epp_image_controller(file)


@router.post("/api/epp/detect-frame")
async def epp_detect_frame(file: UploadFile = File(...)):
    return await detect_epp_frame_controller(file)

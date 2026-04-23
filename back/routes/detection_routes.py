from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from back.controllers.detection_controller import detect_frame_controller, detect_image_controller

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/api/detect-image")
async def detect_image(file: UploadFile = File(...)):
    return await detect_image_controller(file)


@router.post("/api/detect-frame")
async def detect_frame(file: UploadFile = File(...)):
    return await detect_frame_controller(file)

from __future__ import annotations

from fastapi import UploadFile

from back.services.detection_service import detect_frame, detect_image


async def detect_image_controller(file: UploadFile):
    return await detect_image(file)


async def detect_frame_controller(file: UploadFile):
    return await detect_frame(file)

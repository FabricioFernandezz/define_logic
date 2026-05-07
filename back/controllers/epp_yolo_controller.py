from __future__ import annotations

from fastapi import UploadFile
from fastapi.responses import JSONResponse

from back.services.epp_yolo_service import detect_epp_frame, detect_epp_image


async def detect_epp_image_controller(file: UploadFile):
    result = await detect_epp_image(file)
    return JSONResponse(content=result)


async def detect_epp_frame_controller(file: UploadFile):
    result = await detect_epp_frame(file)
    return JSONResponse(content=result)

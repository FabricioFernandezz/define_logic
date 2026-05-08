from __future__ import annotations

from typing import Optional

from fastapi import UploadFile
from fastapi.responses import JSONResponse

from back.services.epp_yolo_service import (
    detect_epp_frame,
    detect_epp_image,
    get_epp_model_classes,
)


async def get_epp_classes_controller():
    result = get_epp_model_classes()
    return JSONResponse(content=result)


async def detect_epp_image_controller(file: UploadFile):
    result = await detect_epp_image(file)
    return JSONResponse(content=result)


async def detect_epp_frame_controller(
    file: UploadFile,
    zones: Optional[str] = None,
    default_zone_epp: Optional[str] = None,
    default_zone_active: Optional[str] = "true",
    default_zone_require_person: Optional[str] = "false",
):
    result = await detect_epp_frame(
        file,
        zones_raw=zones,
        default_zone_epp_raw=default_zone_epp,
        default_zone_active_raw=default_zone_active,
        default_zone_require_person_raw=default_zone_require_person,
    )
    return JSONResponse(content=result)

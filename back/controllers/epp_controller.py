from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from back.services.epp_service import (
    detect_epp_frame,
    detect_epp_from_url,
    detect_epp_image,
    get_epp_model_classes,
)
from back.services.ai_description_service import generate_detection_description


class GenerateDescriptionRequest(BaseModel):
    image_data_url: str
    detections: List[Dict[str, Any]] = []
    person_count: int = 0
    result: str = ""
    alerting_zones: Optional[List[Dict[str, Any]]] = None
    default_zone_result: Optional[Dict[str, Any]] = None


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


async def detect_epp_ip_frame_controller(
    camera_url: str,
    zones: Optional[str] = None,
    default_zone_epp: Optional[str] = None,
    default_zone_active: Optional[str] = "true",
    default_zone_require_person: Optional[str] = "false",
):
    result = await detect_epp_from_url(
        camera_url,
        zones_raw=zones,
        default_zone_epp_raw=default_zone_epp,
        default_zone_active_raw=default_zone_active,
        default_zone_require_person_raw=default_zone_require_person,
    )
    return JSONResponse(content=result)


async def generate_description_controller(payload: GenerateDescriptionRequest):
    try:
        description = await generate_detection_description(
            image_data_url=payload.image_data_url,
            detections=payload.detections,
            person_count=payload.person_count,
            result=payload.result,
            alerting_zones=payload.alerting_zones,
            default_zone_result=payload.default_zone_result,
        )
        return JSONResponse(content={"description": description})
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generando descripción: {exc}") from exc

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from back.controllers.epp_controller import (
    GenerateDescriptionRequest,
    detect_epp_frame_controller,
    detect_epp_image_controller,
    detect_epp_ip_frame_controller,
    generate_description_controller,
    get_epp_classes_controller,
)
from back.controllers.zone_config_controller import (
    get_zone_config_controller,
    save_zone_config_controller,
)
from back.models.zone_config import ZoneConfigSave

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


@router.post("/api/epp/detect-ip-frame")
async def epp_detect_ip_frame(
    camera_url: str = Form(...),
    zones: str = Form(default=None),
    default_zone_epp: str = Form(default=None),
    default_zone_active: str = Form(default="true"),
    default_zone_require_person: str = Form(default="false"),
):
    return await detect_epp_ip_frame_controller(camera_url, zones, default_zone_epp, default_zone_active, default_zone_require_person)


@router.post("/api/epp/generate-description")
async def epp_generate_description(payload: GenerateDescriptionRequest):
    return await generate_description_controller(payload)


@router.get("/api/epp/zone-config")
async def epp_get_zone_config():
    return await get_zone_config_controller()


@router.post("/api/epp/zone-config")
async def epp_save_zone_config(payload: ZoneConfigSave):
    return await save_zone_config_controller(payload)

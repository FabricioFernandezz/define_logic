from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from back.models.zone_config import ZoneConfigSave
from back.services.zone_config_service import get_zone_config, save_zone_config


async def get_zone_config_controller(industry_id: int):
    try:
        result = get_zone_config(industry_id)
        return JSONResponse(content=result)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def save_zone_config_controller(payload: ZoneConfigSave, industry_id: int):
    try:
        result = save_zone_config(
            zones=payload.zones,
            default_zone_epp=payload.defaultZoneEpp,
            default_zone_active=payload.defaultZoneActive,
            default_zone_require_person=payload.defaultZoneRequirePerson,
            industry_id=industry_id,
        )
        return JSONResponse(content=result)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

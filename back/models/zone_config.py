from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel


class ZoneConfigSave(BaseModel):
    zones: List[Any] = []
    defaultZoneEpp: List[str] = []
    defaultZoneActive: bool = True
    defaultZoneRequirePerson: bool = False

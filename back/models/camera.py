from __future__ import annotations

from pydantic import BaseModel, Field


class CameraCreate(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=200)
    url: str = Field(..., min_length=1, max_length=2000)
    activa: bool = True


class CameraUpdate(BaseModel):
    nombre: str | None = Field(default=None, min_length=1, max_length=200)
    url: str | None = Field(default=None, min_length=1, max_length=2000)
    activa: bool | None = None

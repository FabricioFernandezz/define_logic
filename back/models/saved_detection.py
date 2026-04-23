from __future__ import annotations

from pydantic import BaseModel, Field


class SavedDetectionCreate(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=200)
    imagen: str = Field(..., min_length=1)
    descripcion: str | None = None

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class RegisterOwnerRequest(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    industriaNombre: str = Field(..., min_length=1, max_length=200)


class RegisterMemberRequest(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class AllowedEmailRequest(BaseModel):
    email: EmailStr

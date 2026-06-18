from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

from back.models.auth import (
    AllowedEmailRequest,
    LoginRequest,
    RegisterMemberRequest,
    RegisterOwnerRequest,
)
from back.services.auth_service import create_access_token
from back.services.user_service import (
    AuthError,
    add_allowed_email,
    authenticate,
    list_allowed_emails,
    register_member,
    register_owner,
    remove_allowed_email,
)


def _token_response(user: Dict[str, Any]) -> Dict[str, Any]:
    return {"token": create_access_token(user), "user": user}


def register_owner_controller(payload: RegisterOwnerRequest) -> Dict[str, Any]:
    try:
        user = register_owner(
            nombre=payload.nombre,
            email=payload.email,
            password=payload.password,
            industria_nombre=payload.industriaNombre,
        )
        return _token_response(user)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def register_member_controller(payload: RegisterMemberRequest) -> Dict[str, Any]:
    try:
        user = register_member(
            nombre=payload.nombre,
            email=payload.email,
            password=payload.password,
        )
        return _token_response(user)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def login_controller(payload: LoginRequest) -> Dict[str, Any]:
    try:
        user = authenticate(email=payload.email, password=payload.password)
        return _token_response(user)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def list_allowed_emails_controller(industry_id: int):
    try:
        return list_allowed_emails(industry_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def add_allowed_email_controller(industry_id: int, payload: AllowedEmailRequest):
    try:
        return add_allowed_email(industry_id, payload.email)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def remove_allowed_email_controller(industry_id: int, allowed_id: int):
    try:
        remove_allowed_email(industry_id, allowed_id)
        return {"ok": True}
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

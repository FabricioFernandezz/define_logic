from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from back.controllers.auth_controller import (
    add_allowed_email_controller,
    list_allowed_emails_controller,
    login_controller,
    register_member_controller,
    register_owner_controller,
    remove_allowed_email_controller,
)
from back.dependencies import get_current_user, require_owner
from back.models.auth import (
    AllowedEmailRequest,
    LoginRequest,
    RegisterMemberRequest,
    RegisterOwnerRequest,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register-owner")
def register_owner(payload: RegisterOwnerRequest):
    return register_owner_controller(payload)


@router.post("/register")
def register_member(payload: RegisterMemberRequest):
    return register_member_controller(payload)


@router.post("/login")
def login(payload: LoginRequest):
    return login_controller(payload)


@router.get("/me")
def me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return current_user


@router.get("/allowed-emails")
def list_allowed_emails(current_user: Dict[str, Any] = Depends(require_owner)):
    return list_allowed_emails_controller(current_user["industry_id"])


@router.post("/allowed-emails")
def add_allowed_email(
    payload: AllowedEmailRequest,
    current_user: Dict[str, Any] = Depends(require_owner),
):
    return add_allowed_email_controller(current_user["industry_id"], payload)


@router.delete("/allowed-emails/{allowed_id}")
def remove_allowed_email(
    allowed_id: int,
    current_user: Dict[str, Any] = Depends(require_owner),
):
    return remove_allowed_email_controller(current_user["industry_id"], allowed_id)

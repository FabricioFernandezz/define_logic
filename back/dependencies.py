from __future__ import annotations

from typing import Any, Dict

import jwt
from fastapi import Depends, Header, HTTPException

from back.services.auth_service import decode_access_token
from back.services.user_service import get_user_by_id


def get_current_user(authorization: str = Header(default=None)) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="No autenticado")

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Sesión expirada")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token inválido")

    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Token inválido")

    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")

    return user


def require_owner(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if current_user.get("rol") != "owner":
        raise HTTPException(status_code=403, detail="Solo el encargado dueño puede hacer esto")
    return current_user

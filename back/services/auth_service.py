from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import bcrypt
import jwt

_ALGORITHM = "HS256"


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise RuntimeError("JWT_SECRET no configurada. Define JWT_SECRET en back/config/.env")
    return secret


def _expire_minutes() -> int:
    try:
        return int(os.getenv("JWT_EXPIRE_MINUTES", "720"))
    except ValueError:
        return 720


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user: Dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "industry_id": user["industry_id"],
        "rol": user["rol"],
        "iat": now,
        "exp": now + timedelta(minutes=_expire_minutes()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Devuelve el payload o lanza jwt.PyJWTError si invalido/expirado."""
    return jwt.decode(token, _jwt_secret(), algorithms=[_ALGORITHM])

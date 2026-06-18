from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from back.config.database import get_db_engine, init_database
from back.services.auth_service import hash_password, verify_password


class AuthError(RuntimeError):
    """Error de negocio de autenticacion (mensaje apto para el usuario)."""


def _user_public(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "email": row["email"],
        "nombre": row["nombre"],
        "rol": row["rol"],
        "industry_id": row["industry_id"],
    }


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    init_database()
    engine = get_db_engine()
    with engine.begin() as connection:
        row = (
            connection.execute(
                text(
                    "SELECT id, email, nombre, rol, industry_id FROM users WHERE id = :id"
                ),
                {"id": user_id},
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


def register_owner(nombre: str, email: str, password: str, industria_nombre: str) -> Dict[str, Any]:
    """Crea una industria nueva y su usuario owner."""
    init_database()

    nombre = nombre.strip()
    email = email.strip().lower()
    industria_nombre = industria_nombre.strip()
    if not nombre or not email or not password or not industria_nombre:
        raise AuthError("Todos los campos son obligatorios")

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            existing = connection.execute(
                text("SELECT id FROM users WHERE email = :email"), {"email": email}
            ).first()
            if existing is not None:
                raise AuthError("Ese email ya esta registrado")

            industry_id = connection.execute(
                text("INSERT INTO industries (nombre) VALUES (:n) RETURNING id"),
                {"n": industria_nombre},
            ).scalar_one()

            user = (
                connection.execute(
                    text(
                        """
                        INSERT INTO users (email, password_hash, nombre, rol, industry_id)
                        VALUES (:email, :ph, :nombre, 'owner', :iid)
                        RETURNING id, email, nombre, rol, industry_id
                        """
                    ),
                    {
                        "email": email,
                        "ph": hash_password(password),
                        "nombre": nombre,
                        "iid": industry_id,
                    },
                )
                .mappings()
                .one()
            )

            connection.execute(
                text("UPDATE industries SET owner_user_id = :uid WHERE id = :iid"),
                {"uid": user["id"], "iid": industry_id},
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo registrar la industria: {exc}") from exc

    return _user_public(dict(user))


def register_member(nombre: str, email: str, password: str) -> Dict[str, Any]:
    """Registra un encargado cuyo mail debe estar en la whitelist de alguna industria."""
    init_database()

    nombre = nombre.strip()
    email = email.strip().lower()
    if not nombre or not email or not password:
        raise AuthError("Todos los campos son obligatorios")

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            existing = connection.execute(
                text("SELECT id FROM users WHERE email = :email"), {"email": email}
            ).first()
            if existing is not None:
                raise AuthError("Ese email ya esta registrado")

            invite = (
                connection.execute(
                    text(
                        "SELECT id, industry_id, used FROM industry_allowed_emails WHERE email = :email"
                    ),
                    {"email": email},
                )
                .mappings()
                .first()
            )
            if invite is None:
                raise AuthError("Tu email no fue invitado a ninguna industria. Pide al encargado que te agregue.")
            if invite["used"]:
                raise AuthError("Esa invitacion ya fue utilizada")

            user = (
                connection.execute(
                    text(
                        """
                        INSERT INTO users (email, password_hash, nombre, rol, industry_id)
                        VALUES (:email, :ph, :nombre, 'member', :iid)
                        RETURNING id, email, nombre, rol, industry_id
                        """
                    ),
                    {
                        "email": email,
                        "ph": hash_password(password),
                        "nombre": nombre,
                        "iid": invite["industry_id"],
                    },
                )
                .mappings()
                .one()
            )

            connection.execute(
                text("UPDATE industry_allowed_emails SET used = TRUE WHERE id = :id"),
                {"id": invite["id"]},
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo registrar el usuario: {exc}") from exc

    return _user_public(dict(user))


def authenticate(email: str, password: str) -> Dict[str, Any]:
    init_database()

    email = email.strip().lower()
    engine = get_db_engine()
    with engine.begin() as connection:
        row = (
            connection.execute(
                text(
                    "SELECT id, email, nombre, rol, industry_id, password_hash FROM users WHERE email = :email"
                ),
                {"email": email},
            )
            .mappings()
            .first()
        )

    if row is None or not verify_password(password, row["password_hash"]):
        raise AuthError("Email o contraseña incorrectos")

    return _user_public(dict(row))


# --- Whitelist (solo owner) ---

def list_allowed_emails(industry_id: int) -> List[Dict[str, Any]]:
    init_database()
    engine = get_db_engine()
    with engine.begin() as connection:
        rows = (
            connection.execute(
                text(
                    """
                    SELECT id, email, used, created_at
                    FROM industry_allowed_emails
                    WHERE industry_id = :iid
                    ORDER BY id DESC
                    """
                ),
                {"iid": industry_id},
            )
            .mappings()
            .all()
        )
    return [
        {
            "id": r["id"],
            "email": r["email"],
            "used": bool(r["used"]),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


def add_allowed_email(industry_id: int, email: str) -> Dict[str, Any]:
    init_database()
    email = email.strip().lower()
    if not email:
        raise AuthError("El email es obligatorio")

    engine = get_db_engine()
    with engine.begin() as connection:
        clash = (
            connection.execute(
                text("SELECT industry_id FROM industry_allowed_emails WHERE email = :email"),
                {"email": email},
            )
            .mappings()
            .first()
        )
        if clash is not None:
            if clash["industry_id"] == industry_id:
                raise AuthError("Ese email ya esta en tu lista")
            raise AuthError("Ese email ya pertenece a otra industria")

        user_clash = connection.execute(
            text("SELECT id FROM users WHERE email = :email"), {"email": email}
        ).first()
        if user_clash is not None:
            raise AuthError("Ese email ya tiene una cuenta")

        row = (
            connection.execute(
                text(
                    """
                    INSERT INTO industry_allowed_emails (industry_id, email)
                    VALUES (:iid, :email)
                    RETURNING id, email, used, created_at
                    """
                ),
                {"iid": industry_id, "email": email},
            )
            .mappings()
            .one()
        )
    return {
        "id": row["id"],
        "email": row["email"],
        "used": bool(row["used"]),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


def remove_allowed_email(industry_id: int, allowed_id: int) -> None:
    init_database()
    engine = get_db_engine()
    with engine.begin() as connection:
        result = connection.execute(
            text(
                "DELETE FROM industry_allowed_emails WHERE id = :id AND industry_id = :iid AND used = FALSE"
            ),
            {"id": allowed_id, "iid": industry_id},
        )
        if result.rowcount == 0:
            raise AuthError("No se pudo eliminar (no existe, no es tuyo, o ya fue usado)")

from __future__ import annotations

import json
from typing import Any, Dict

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from back.config.database import get_db_engine, init_database

_CONFIG_ID = 1


def get_zone_config() -> Dict[str, Any]:
    init_database()

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT zones, default_zone_epp, default_zone_active, default_zone_require_person
                        FROM epp_zone_config
                        WHERE id = :id
                        """
                    ),
                    {"id": _CONFIG_ID},
                )
                .mappings()
                .first()
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo obtener configuración de zonas: {exc}") from exc

    if row is None:
        return {
            "zones": [],
            "defaultZoneEpp": [],
            "defaultZoneActive": True,
            "defaultZoneRequirePerson": False,
        }

    return {
        "zones": json.loads(row["zones"]),
        "defaultZoneEpp": json.loads(row["default_zone_epp"]),
        "defaultZoneActive": bool(row["default_zone_active"]),
        "defaultZoneRequirePerson": bool(row["default_zone_require_person"]),
    }


def save_zone_config(
    zones: list,
    default_zone_epp: list,
    default_zone_active: bool,
    default_zone_require_person: bool,
) -> Dict[str, Any]:
    init_database()

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO epp_zone_config (id, zones, default_zone_epp, default_zone_active, default_zone_require_person)
                    VALUES (:id, :zones, :default_zone_epp, :default_zone_active, :default_zone_require_person)
                    ON CONFLICT (id) DO UPDATE SET
                        zones = EXCLUDED.zones,
                        default_zone_epp = EXCLUDED.default_zone_epp,
                        default_zone_active = EXCLUDED.default_zone_active,
                        default_zone_require_person = EXCLUDED.default_zone_require_person
                    """
                ),
                {
                    "id": _CONFIG_ID,
                    "zones": json.dumps(zones),
                    "default_zone_epp": json.dumps(default_zone_epp),
                    "default_zone_active": default_zone_active,
                    "default_zone_require_person": default_zone_require_person,
                },
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"No se pudo guardar configuración de zonas: {exc}") from exc

    return {
        "zones": zones,
        "defaultZoneEpp": default_zone_epp,
        "defaultZoneActive": default_zone_active,
        "defaultZoneRequirePerson": default_zone_require_person,
    }

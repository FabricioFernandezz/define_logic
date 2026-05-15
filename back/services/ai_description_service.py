from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "config" / ".env")


def _build_context(
    detections: List[Dict[str, Any]],
    person_count: int,
    result: str,
    alerting_zones: Optional[List[Dict[str, Any]]] = None,
    default_zone_result: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [f"Personas detectadas: {person_count}"]
    if detections:
        labels = [f"{d['label']} ({d['confidence']*100:.0f}%)" for d in detections]
        lines.append(f"EPP detectado: {', '.join(labels)}")
    else:
        lines.append("EPP detectado: ninguno")

    if alerting_zones:
        zone_names = [z.get("label") or z.get("zoneId", "") for z in alerting_zones]
        lines.append(f"Zonas en alerta: {', '.join(zone_names)}")

    if default_zone_result and not default_zone_result.get("compliant", True):
        missing = default_zone_result.get("missingEpp", [])
        if missing:
            lines.append(f"Zona general faltante: {', '.join(missing)}")

    lines.append(f"Resultado: {result}")
    return "\n".join(lines)


async def generate_detection_description(
    image_data_url: str,
    detections: List[Dict[str, Any]],
    person_count: int,
    result: str,
    alerting_zones: Optional[List[Dict[str, Any]]] = None,
    default_zone_result: Optional[Dict[str, Any]] = None,
) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no configurada en back/config/.env")

    client = genai.Client(api_key=api_key)
    context = _build_context(detections, person_count, result, alerting_zones, default_zone_result)

    if "," in image_data_url:
        header, b64 = image_data_url.split(",", 1)
        media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
    else:
        b64 = image_data_url
        media_type = "image/jpeg"

    image_bytes = base64.b64decode(b64)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=(
                "Eres un sistema de registro de seguridad industrial. "
                "Describe en máximo 30 palabras qué se detectó, qué EPP falta o está presente y en qué zona. "
                "Solo español. Sin saludos ni explicaciones extra."
            )
        ),
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=media_type),
            types.Part.from_text(text=f"Datos del modelo:\n{context}"),
        ],
    )

    return response.text.strip()

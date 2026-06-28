from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from back.services.auth_service import decode_access_token
from back.services.epp_service import IpCameraStream, process_frame_bytes

router = APIRouter()


def _config_to_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    zones = cfg.get("zones")
    default_epp = cfg.get("defaultZoneEpp")
    mode = cfg.get("mode")
    camera_id = cfg.get("camera_url") if mode == "ip" else ("webcam" if mode == "webcam" else None)
    return {
        "zones_raw": json.dumps(zones) if zones else None,
        "default_zone_epp_raw": json.dumps(default_epp) if default_epp else None,
        "default_zone_active_raw": "true" if cfg.get("defaultZoneActive", True) else "false",
        "default_zone_require_person_raw": "true" if cfg.get("defaultZoneRequirePerson", False) else "false",
        "camera_id": camera_id,
    }


@router.websocket("/ws/epp/detect")
async def epp_detect_ws(websocket: WebSocket) -> None:
    """Persistent detection channel.

    Auth: JWT passed as the `token` query param (browsers can't set headers on WS).
    Protocol (client -> server):
      - text  {"type":"config", "mode":"webcam"|"ip", "camera_url":..., zones, default*}
      - text  {"type":"stop"}                          -> stop IP push loop
      - bytes <jpeg frame>                             -> webcam frame to analyze
    Server -> client: text JSON detection payloads (same shape as the REST endpoints).
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        decode_access_token(token)
    except jwt.PyJWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    config: Dict[str, Any] = {}
    ip_task: Optional[asyncio.Task] = None

    async def ip_loop(camera_url: str) -> None:
        # Server-driven push: ONE persistent connection to the camera; read + infer
        # + send until cancelled or the socket drops. Reconnects with backoff on error.
        stream = IpCameraStream(camera_url)
        stream.start()
        try:
            while True:
                try:
                    # Always grab the freshest frame; the reader thread drops the
                    # backlog so latency stays flat. Inference itself paces the loop.
                    frame_bytes = await stream.latest_frame()
                    payload = await asyncio.to_thread(
                        process_frame_bytes,
                        frame_bytes,
                        **_config_to_kwargs(config),
                        always_annotate=True,
                    )
                    await websocket.send_text(json.dumps(payload))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — surface to client, keep loop alive
                    try:
                        await websocket.send_text(json.dumps({"error": str(exc)[:200]}))
                    except Exception:
                        return
                    await asyncio.sleep(0.5)
        finally:
            await stream.close()

    def _stop_ip() -> None:
        nonlocal ip_task
        if ip_task and not ip_task.done():
            ip_task.cancel()
        ip_task = None

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text = message.get("text")
            data_bytes = message.get("bytes")

            if text is not None:
                try:
                    data = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    continue
                mtype = data.get("type")
                if mtype == "config":
                    config.update({k: v for k, v in data.items() if k != "type"})
                    if data.get("mode") == "ip" and data.get("camera_url"):
                        _stop_ip()
                        ip_task = asyncio.create_task(ip_loop(data["camera_url"]))
                    else:
                        _stop_ip()
                elif mtype == "stop":
                    _stop_ip()
            elif data_bytes:
                # webcam frame — run inference off the event loop
                try:
                    payload = await asyncio.to_thread(
                        process_frame_bytes,
                        data_bytes,
                        **_config_to_kwargs(config),
                        always_annotate=False,
                    )
                    await websocket.send_text(json.dumps(payload))
                except Exception as exc:  # noqa: BLE001
                    await websocket.send_text(json.dumps({"error": str(exc)[:200]}))
    except WebSocketDisconnect:
        pass
    finally:
        _stop_ip()

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

EPP_MODEL_PATH = project_root / "ml" / "runs" / "yolo26_epp" / "best.onnx"
EPP_CONF_THRESHOLD = 0.35

_epp_model = None

NON_COMPLIANT_KEYWORDS = {"no_", "sin_", "without_", "no-", "sin-", "without-"}


def _is_non_compliant(label: str) -> bool:
    lower = label.lower()
    return any(lower.startswith(kw) or f"_{kw.strip('_')}" in lower for kw in NON_COMPLIANT_KEYWORDS) or \
           any(kw in lower for kw in ("no_helmet", "sin_casco", "no_hardhat", "without_helmet",
                                       "no helmet", "sin casco", "no casco"))


def init_epp_model() -> None:
    global _epp_model
    if _epp_model is not None:
        return

    if not EPP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo EPP no encontrado: {EPP_MODEL_PATH}. "
            "Verifica que el archivo best.onnx existe en ml/runs/yolo26_epp/"
        )

    try:
        from ultralytics import YOLO
        _epp_model = YOLO(str(EPP_MODEL_PATH))
    except Exception as exc:
        raise RuntimeError(f"Error cargando modelo EPP ONNX: {exc}") from exc


def _run_inference(image_rgb: np.ndarray) -> List[Dict[str, Any]]:
    results = _epp_model(image_rgb, conf=EPP_CONF_THRESHOLD, verbose=False)
    detections: List[Dict[str, Any]] = []

    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            label = names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box)
            detections.append(
                {
                    "id": i,
                    "bbox_pixels": [x1, y1, x2, y2],
                    "label": label,
                    "confidence": float(conf),
                    "isCompliant": not _is_non_compliant(label),
                }
            )

    return detections


def _annotate(image_rgb: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_pixels"]
        color = (0, 200, 0) if det["isCompliant"] else (0, 0, 255)
        label_text = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_bgr,
            label_text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return image_bgr


def _encode(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise RuntimeError("No se pudo codificar imagen EPP")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _build_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    compliant = sum(1 for d in detections if d["isCompliant"])
    non_compliant = len(detections) - compliant
    if non_compliant > 0:
        result = "mixto" if compliant > 0 else "no cumple"
    elif compliant > 0:
        result = "cumple"
    else:
        result = "sin detecciones"

    avg_conf = (
        float(round(sum(d["confidence"] for d in detections) / len(detections), 2))
        if detections
        else 0.0
    )
    return {
        "compliantCount": compliant,
        "nonCompliantCount": non_compliant,
        "result": result,
        "confidence": avg_conf,
    }


async def detect_epp_image(file: UploadFile) -> Dict[str, Any]:
    init_epp_model()
    started_at = time.perf_counter()

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="La imagen está vacía")

    np_buffer = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections = _run_inference(image_rgb)
    summary = _build_summary(detections)
    annotated = _annotate(image_rgb, detections)
    annotated_url = _encode(annotated)
    processing_time_ms = int((time.perf_counter() - started_at) * 1000)

    return {
        "modelName": "yolo26_epp",
        "processingTimeMs": processing_time_ms,
        "imageSize": {"width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])},
        "detections": detections,
        "summary": summary,
        "annotatedImage": annotated_url,
        "sourceFile": file.filename,
    }


async def detect_epp_frame(file: UploadFile) -> Dict[str, Any]:
    init_epp_model()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Frame vacío")

    np_buffer = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar el frame")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections = _run_inference(image_rgb)

    if not detections:
        return {"detections": [], "alert": False}

    alert = any(not d["isCompliant"] for d in detections)
    payload: Dict[str, Any] = {"detections": detections, "alert": alert}

    annotated = _annotate(image_rgb, detections)
    payload["annotated_frame"] = _encode(annotated)

    return payload

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from config import (  # noqa: E402
    HELMET_HEAD_REGION_RATIO,
    ViT_INFERENCE_DEVICE,
    ViT_MODEL_NAME,
    ViT_MODEL_PATH,
    ViT_THRESHOLD,
    YOLO_CONF_THRESHOLD,
    YOLO_INFERENCE_DEVICE,
    YOLO_MODEL_PATH,
)
from ml.models.epp_detectors import HelmetDetector  # noqa: E402
from ml.models.person_detector import PersonDetector  # noqa: E402

app = FastAPI(title="Helmet Vision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

person_detector: PersonDetector | None = None
helmet_detector: HelmetDetector | None = None


def _resolve_helmet_model_path() -> str:
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path(ViT_MODEL_PATH),
        project_root / ViT_MODEL_PATH,
        project_root / "ml" / ViT_MODEL_PATH,
        project_root / "ml" / "models" / "models" / "vit_epp_best.pt",
    ]

    for model_path in candidates:
        if model_path.exists():
            return str(model_path)

    model_path = project_root / "ml" / "models" / "models" / "vit_epp_best.pt"
    raise FileNotFoundError(
        f"No se encontro el modelo final requerido: {model_path}. "
        "Ejecuta primero el entrenamiento principal para generar vit_epp_best.pt"
    )


def _init_detectors() -> None:
    global person_detector, helmet_detector

    if person_detector is None:
        person_detector = PersonDetector(
            model_name=YOLO_MODEL_PATH,
            device=YOLO_INFERENCE_DEVICE,
            conf_threshold=YOLO_CONF_THRESHOLD,
        )

    if helmet_detector is None:
        helmet_detector = HelmetDetector(
            model_path=_resolve_helmet_model_path(),
            model_name=ViT_MODEL_NAME,
            device=ViT_INFERENCE_DEVICE,
            threshold=ViT_THRESHOLD,
        )


def _build_summary(persons: List[Dict[str, Any]]) -> Dict[str, Any]:
    helmet_count = sum(1 for person in persons if person["helmetDetected"])
    no_helmet_count = len(persons) - helmet_count
    if no_helmet_count > 0:
        result = "mixto" if helmet_count > 0 else "sin casco"
    else:
        result = "con casco"

    confidence_values = [person["confidence"] for person in persons]
    confidence = float(round(sum(confidence_values) / len(confidence_values), 2)) if confidence_values else 0.0

    return {
        "helmetCount": helmet_count,
        "noHelmetCount": no_helmet_count,
        "result": result,
        "confidence": confidence,
    }


def _annotate_image(image_rgb: np.ndarray, persons: List[Dict[str, Any]]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for person in persons:
        x1, y1, x2, y2 = person["bbox_pixels"]
        is_helmet = person["helmetDetected"]
        color = (0, 200, 0) if is_helmet else (0, 0, 255)
        label = f"{person['label']} {person['confidence']:.2f}"

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_bgr,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return image_bgr


def _encode_image_to_data_url(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise RuntimeError("No se pudo codificar la imagen resultante")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@app.on_event("startup")
def startup_event() -> None:
    _init_detectors()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/detect-image")
async def detect_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    _init_detectors()
    started_at = time.perf_counter()

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="La imagen esta vacia")

    np_buffer = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections = person_detector.detect(image_rgb, conf=YOLO_CONF_THRESHOLD)

    persons: List[Dict[str, Any]] = []
    for idx, detection in enumerate(detections):
        bbox = detection["bbox_pixels"]
        x1, y1, x2, y2 = bbox
        person_h = max(1, y2 - y1)
        head_y2 = y1 + int(person_h * HELMET_HEAD_REGION_RATIO)
        head_bbox = (x1, y1, x2, head_y2)

        crop_rgb = person_detector.crop_person(image_rgb, head_bbox, padding=12)
        if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.shape[0] < 20 or crop_rgb.shape[1] < 20:
            crop_rgb = person_detector.crop_person(image_rgb, bbox, padding=10)

        result = helmet_detector.detect(crop_rgb)
        persons.append(
            {
                "personId": idx,
                "bbox_pixels": list(map(int, bbox)),
                "bbox_norm": [float(v) for v in detection["bbox_norm"]],
                "detection_conf": float(detection["confidence"]),
                "helmetDetected": bool(result.is_compliant),
                "label": result.label,
                "confidence": float(result.confidence),
            }
        )

    summary = _build_summary(persons)
    annotated = _annotate_image(image_rgb, persons)
    annotated_data_url = _encode_image_to_data_url(annotated)
    processing_time_ms = int((time.perf_counter() - started_at) * 1000)

    return {
        "modelName": Path(ViT_MODEL_PATH).name if Path(ViT_MODEL_PATH).exists() else "unknown",
        "processingTimeMs": processing_time_ms,
        "imageSize": {"width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])},
        "detections": persons,
        "summary": summary,
        "annotatedImage": annotated_data_url,
        "sourceFile": file.filename,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

"""Inferencia estatica: YOLO (personas) + ViT (casco).

Flujo:
- Carga una imagen estatica.
- YOLO detecta personas.
- ViT clasifica casco en cada crop de persona.
- Reporta y dibuja el conteo total de cascos detectados.
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2

from config import (
    HELMET_HEAD_REGION_RATIO,
    STATIC_IMAGE_SOURCE,
    ViT_MODEL_NAME,
    ViT_MODEL_PATH,
    ViT_INFERENCE_DEVICE,
    ViT_THRESHOLD,
    YOLO_CONF_THRESHOLD,
    YOLO_INFERENCE_DEVICE,
    YOLO_MODEL_PATH,
)
from models.epp_detectors import HelmetDetector
from models.person_detector import PersonDetector


def _resolve_image_path() -> Path:
    """Obtiene la ruta de entrada desde argumento o desde config.py."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    return Path(STATIC_IMAGE_SOURCE).expanduser().resolve()


def _resolve_model_path() -> str:
    """Usa exclusivamente el checkpoint final entrenado (sin fallbacks)."""
    full_model = Path(ViT_MODEL_PATH)
    if full_model.exists():
        return str(full_model)
    raise FileNotFoundError(
        f"No se encontro el modelo final requerido: {full_model}. "
        "Ejecuta primero el entrenamiento principal para generar vit_epp_best.pt"
    )


def main() -> None:
    """Ejecuta inferencia de casco por persona sobre una imagen."""
    print("=" * 70)
    print("DEFINE LOGIC - DETECCION DE CASCO CON VIT")
    print("=" * 70)

    image_path = _resolve_image_path()
    if not image_path.exists():
        raise FileNotFoundError(
            f"No se encontro la imagen de entrada: {image_path}\n"
            f"Pasa una ruta valida como argumento o crea {STATIC_IMAGE_SOURCE}"
        )

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"No se pudo cargar la imagen: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    person_detector = PersonDetector(
        model_name=YOLO_MODEL_PATH,
        device=YOLO_INFERENCE_DEVICE,
        conf_threshold=YOLO_CONF_THRESHOLD,
    )
    helmet_detector = HelmetDetector(
        model_path=_resolve_model_path(),
        model_name=ViT_MODEL_NAME,
        device=ViT_INFERENCE_DEVICE,
        threshold=ViT_THRESHOLD,
    )

    detections = person_detector.detect(image_rgb, conf=YOLO_CONF_THRESHOLD)

    annotated = image_bgr.copy()
    helmets_detected = 0
    people_detected = len(detections)

    for idx, det in enumerate(detections):
        bbox = det["bbox_pixels"]
        x1, y1, x2, y2 = bbox

        # Para casco, la señal más útil está en la parte superior del bbox (cabeza/hombros).
        # Esto reduce falsos positivos cuando el cuerpo completo introduce ruido.
        person_h = max(1, y2 - y1)
        head_y2 = y1 + int(person_h * HELMET_HEAD_REGION_RATIO)
        head_bbox = (x1, y1, x2, head_y2)
        crop_rgb = person_detector.crop_person(image_rgb, head_bbox, padding=12)

        # Fallback defensivo para personas muy pequeñas.
        if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.shape[0] < 20 or crop_rgb.shape[1] < 20:
            crop_rgb = person_detector.crop_person(image_rgb, bbox, padding=10)

        result = helmet_detector.detect(crop_rgb)

        if result.is_compliant:
            helmets_detected += 1

        color = (0, 200, 0) if result.is_compliant else (0, 0, 255)
        label_text = f"P{idx} {result.label} {result.confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label_text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    summary_text = f"Cascos: {helmets_detected}/{people_detected}"
    cv2.putText(
        annotated,
        summary_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    output_dir = Path("runs/inference")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_helmet.jpg"
    cv2.imwrite(str(output_path), annotated)

    print(f"[OK] Imagen de entrada: {image_path}")
    print(f"[OK] Modelo usado: {_resolve_model_path()}")
    print(f"[OK] Dispositivo YOLO: {YOLO_INFERENCE_DEVICE}")
    print(f"[OK] Dispositivo ViT: {ViT_INFERENCE_DEVICE}")
    print(f"[OK] Personas detectadas (YOLO): {people_detected}")
    print(f"[OK] Cascos detectados (ViT): {helmets_detected}")
    print(f"[OK] Salida guardada en: {output_path}")


if __name__ == "__main__":
    main()

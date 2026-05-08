from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

EPP_MODEL_PATH = project_root / "ml" / "runs" / "yolo26_epp" / "best.onnx"
EPP_CONF_THRESHOLD = 0.35

_epp_model = None

NON_COMPLIANT_KEYWORDS = {"no_", "sin_", "without_", "no-", "sin-", "without-"}

PERSON_LABELS: frozenset = frozenset({"person", "worker", "human", "people", "persona", "trabajador"})


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
    person_count = sum(1 for d in detections if d["label"].lower() in PERSON_LABELS)
    display_detections = [d for d in detections if d["label"].lower() not in PERSON_LABELS]
    summary = _build_summary(display_detections)
    annotated = _annotate(image_rgb, display_detections)
    annotated_url = _encode(annotated)
    processing_time_ms = int((time.perf_counter() - started_at) * 1000)

    return {
        "modelName": "yolo26_epp",
        "processingTimeMs": processing_time_ms,
        "imageSize": {"width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])},
        "detections": display_detections,
        "personCount": person_count,
        "summary": summary,
        "annotatedImage": annotated_url,
        "sourceFile": file.filename,
    }


def get_epp_model_classes() -> Dict[str, Any]:
    init_epp_model()
    all_classes = sorted(_epp_model.names.values())
    # Exclude person labels — zone compliance uses them internally, not as selectable EPP
    compliant = [c for c in all_classes if not _is_non_compliant(c) and c.lower() not in PERSON_LABELS]
    return {"all": all_classes, "compliant": compliant}


def _det_center_in_zone(det: Dict[str, Any], bbox: Dict, img_w: int, img_h: int) -> bool:
    zx1 = bbox.get("x", 0) * img_w
    zy1 = bbox.get("y", 0) * img_h
    zx2 = zx1 + bbox.get("w", 0) * img_w
    zy2 = zy1 + bbox.get("h", 0) * img_h
    cx = (det["bbox_pixels"][0] + det["bbox_pixels"][2]) / 2
    cy = (det["bbox_pixels"][1] + det["bbox_pixels"][3]) / 2
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def _missing_epp(zone_dets: List[Dict[str, Any]], required: List[str]) -> List[str]:
    return [
        req for req in required
        if not any(req in d["label"].lower() and d["isCompliant"] for d in zone_dets)
    ]


def _zone_compliance_for_dets(
    zone_dets: List[Dict[str, Any]],
    required: List[str],
    require_person: bool = False,
) -> tuple[List[str], List[str]]:
    """Returns (missing, violations).
    require_person=True  → only check EPP if a person is detected in the zone.
    require_person=False → check EPP items directly regardless of person presence.
    missing    = required EPP not compliantly present.
    violations = required EPP where non-compliant version (no_X) was explicitly detected.
    """
    if require_person and not any(d["label"].lower() in PERSON_LABELS for d in zone_dets):
        return [], []  # person required but none in zone → skip

    missing: List[str] = []
    violations: List[str] = []
    for req in required:
        compliant_found = any(req in d["label"].lower() and d["isCompliant"] for d in zone_dets)
        noncompliant_found = any(req in d["label"].lower() and not d["isCompliant"] for d in zone_dets)
        if not compliant_found:
            missing.append(req)
        if noncompliant_found:
            violations.append(req)
    return missing, violations


def _check_zone_compliance(
    detections: List[Dict[str, Any]],
    zones: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    results = []
    for zone in zones:
        active = zone.get("active", True)
        if not active:
            results.append({
                "zoneId": zone.get("id", ""),
                "label": zone.get("label", ""),
                "active": False,
                "detectionCount": 0,
                "missingEpp": [],
                "violatingEpp": [],
                "compliant": True,
                "hasRequired": False,
            })
            continue

        bbox = zone.get("bbox", {})
        required = [e.lower() for e in zone.get("required_epp", [])]
        require_person = bool(zone.get("require_person", False))
        zone_dets = [d for d in detections if _det_center_in_zone(d, bbox, img_w, img_h)]
        missing, violations = _zone_compliance_for_dets(zone_dets, required, require_person)
        results.append({
            "zoneId": zone.get("id", ""),
            "label": zone.get("label", ""),
            "active": True,
            "detectionCount": len(zone_dets),
            "missingEpp": missing,
            "violatingEpp": violations,
            "compliant": len(missing) == 0 and len(violations) == 0,
            "hasRequired": len(required) > 0,
        })
    return results


def _check_default_zone(
    detections: List[Dict[str, Any]],
    zones: List[Dict[str, Any]],
    default_epp: List[str],
    default_active: bool,
    default_require_person: bool,
    img_w: int,
    img_h: int,
) -> Optional[Dict[str, Any]]:
    if not default_epp and not default_active:
        return None
    if not default_active:
        return {
            "zoneId": "__default__",
            "label": "Zona por defecto",
            "active": False,
            "detectionCount": 0,
            "missingEpp": [],
            "violatingEpp": [],
            "compliant": True,
            "hasRequired": False,
        }
    if not default_epp:
        return None
    outside_dets = [
        d for d in detections
        if not any(_det_center_in_zone(d, z.get("bbox", {}), img_w, img_h) for z in zones)
    ]
    required = [e.lower() for e in default_epp]
    missing, violations = _zone_compliance_for_dets(outside_dets, required, default_require_person)
    return {
        "zoneId": "__default__",
        "label": "Zona por defecto",
        "active": True,
        "detectionCount": len(outside_dets),
        "missingEpp": missing,
        "violatingEpp": violations,
        "compliant": len(missing) == 0 and len(violations) == 0,
        "hasRequired": True,
    }


async def detect_epp_frame(
    file: UploadFile,
    zones_raw: Optional[str] = None,
    default_zone_epp_raw: Optional[str] = None,
    default_zone_active_raw: Optional[str] = "true",
    default_zone_require_person_raw: Optional[str] = "false",
) -> Dict[str, Any]:
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
    img_h, img_w = image_bgr.shape[:2]

    # Persons are used internally for zone compliance but not shown in the display panel
    display_detections = [d for d in detections if d["label"].lower() not in PERSON_LABELS]

    zones: List[Dict[str, Any]] = []
    if zones_raw:
        try:
            zones = json.loads(zones_raw)
        except (json.JSONDecodeError, TypeError):
            zones = []

    default_epp: List[str] = []
    if default_zone_epp_raw:
        try:
            default_epp = json.loads(default_zone_epp_raw)
        except (json.JSONDecodeError, TypeError):
            default_epp = []

    default_active: bool = (default_zone_active_raw or "true").strip().lower() != "false"
    default_require_person: bool = (default_zone_require_person_raw or "false").strip().lower() == "true"

    zone_results: List[Dict[str, Any]] = []
    default_zone_result: Optional[Dict[str, Any]] = None

    if zones or default_epp or not default_active:
        # Full detections passed so zone compliance can use person labels when require_person=True
        zone_results = _check_zone_compliance(detections, zones, img_w, img_h)
        default_zone_result = _check_default_zone(
            detections, zones, default_epp, default_active, default_require_person, img_w, img_h
        )
        zone_alert = any(not z["compliant"] and z["hasRequired"] for z in zone_results)
        default_alert = bool(default_zone_result and not default_zone_result["compliant"])
        alert = zone_alert or default_alert
    else:
        alert = any(not d["isCompliant"] for d in display_detections) if display_detections else False

    person_count = sum(1 for d in detections if d["label"].lower() in PERSON_LABELS)

    payload: Dict[str, Any] = {
        "detections": display_detections,
        "personCount": person_count,
        "alert": alert,
        "zoneResults": zone_results,
        "defaultZoneResult": default_zone_result,
    }

    if detections or alert:
        annotated = _annotate(image_rgb, display_detections)
        if zone_results or default_zone_result:
            annotated = _annotate_zones_on_frame(annotated, zones, zone_results, img_w, img_h)
        payload["annotated_frame"] = _encode(annotated)

    return payload


def _annotate_zones_on_frame(
    image_bgr: np.ndarray,
    zones: List[Dict[str, Any]],
    zone_results: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    result_by_id = {z["zoneId"]: z for z in zone_results}
    for zone in zones:
        bbox = zone.get("bbox", {})
        x1 = int(bbox.get("x", 0) * img_w)
        y1 = int(bbox.get("y", 0) * img_h)
        x2 = int(x1 + bbox.get("w", 0) * img_w)
        y2 = int(y1 + bbox.get("h", 0) * img_h)
        zr = result_by_id.get(zone.get("id", ""), {})
        compliant = zr.get("compliant", True)
        has_req = zr.get("hasRequired", False)
        color = (0, 200, 0) if (compliant or not has_req) else (0, 0, 255)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        label = zone.get("label", "")
        missing = zr.get("missingEpp", [])
        text = f"{label}" + (f" ✗ {','.join(missing)}" if missing else " ✓")
        cv2.putText(image_bgr, text, (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image_bgr

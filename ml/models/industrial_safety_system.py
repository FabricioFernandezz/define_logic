"""Sistema modular de vision artificial para seguridad industrial.

Flujo por frame:
1. Detectar personas con YOLO.
2. Validar si su centroide esta dentro de una zona ROI poligonal.
3. Si esta dentro, recortar persona y clasificar casco con HelmetDetector.
4. Si resultado es "sin casco", guardar evidencia (crop) para auditoria.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from models.epp_detectors import EPPDetector
from models.person_detector import PersonDetector


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


def person_centroid(bbox: BBox) -> Point:
    """Calcula centroide de una caja delimitadora (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)


def is_centroid_inside_polygon(centroid: Point, polygon: Sequence[Point]) -> bool:
    """Verifica si un punto esta dentro (o borde) de un poligono."""
    if len(polygon) < 3:
        return False
    poly_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, centroid, False) >= 0


def is_person_in_required_zone(bbox: BBox, required_polygon: Sequence[Point]) -> bool:
    """Valida si el centroide de una persona cae dentro de la ROI requerida."""
    centroid = person_centroid(bbox)
    return is_centroid_inside_polygon(centroid, required_polygon)


class IndustrialSafetyVisionSystem:
    """Orquestador principal del flujo YOLO + detector EPP."""

    def __init__(
        self,
        person_detector: PersonDetector,
        epp_detector: EPPDetector,
        required_zone_polygon: Sequence[Point],
        audit_output_dir: str = "runs/audit",
        crop_padding: int = 10,
    ) -> None:
        self.person_detector = person_detector
        self.epp_detector = epp_detector
        self.required_zone_polygon = list(required_zone_polygon)
        self.audit_output_dir = Path(audit_output_dir)
        self.crop_output_dir = self.audit_output_dir.parent / "crops"
        self.compliant_output_dir = self.crop_output_dir / "helmet"
        self.non_compliant_output_dir = self.crop_output_dir / "no_helmet"
        self.crop_padding = crop_padding
        self.audit_output_dir.mkdir(parents=True, exist_ok=True)
        self.compliant_output_dir.mkdir(parents=True, exist_ok=True)
        self.non_compliant_output_dir.mkdir(parents=True, exist_ok=True)

    def _save_crop(self, crop_rgb: np.ndarray, frame_idx: int, person_id: int, compliant: bool) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target_dir = self.compliant_output_dir if compliant else self.non_compliant_output_dir
        prefix = "helmet" if compliant else "no_helmet"
        out_path = target_dir / f"{prefix}_f{frame_idx:06d}_p{person_id}_{ts}.jpg"
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), crop_bgr)
        return str(out_path)

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> Dict:
        """Ejecuta deteccion, verificacion de zona y auditoria por frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Frame vacio o invalido")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = self.person_detector.detect(frame_rgb)

        people_results: List[Dict] = []
        violations: List[Dict] = []

        for person_id, det in enumerate(detections):
            bbox = det["bbox_pixels"]
            centroid = person_centroid(bbox)
            in_zone = is_person_in_required_zone(bbox, self.required_zone_polygon)

            person_result: Dict = {
                "person_id": person_id,
                "bbox_pixels": bbox,
                "centroid": centroid,
                "detection_confidence": det["confidence"],
                "in_required_zone": in_zone,
                "helmet_result": None,
            }

            if in_zone:
                crop_rgb = self.person_detector.crop_person(frame_rgb, bbox, padding=self.crop_padding)
                helmet_result = self.epp_detector.detect(crop_rgb)
                crop_path = self._save_crop(
                    crop_rgb,
                    frame_idx=frame_idx,
                    person_id=person_id,
                    compliant=helmet_result.is_compliant,
                )
                person_result["helmet_result"] = asdict(helmet_result)
                person_result["crop_path"] = crop_path

                if not helmet_result.is_compliant:
                    violations.append(
                        {
                            "frame_idx": frame_idx,
                            "person_id": person_id,
                            "bbox_pixels": bbox,
                            "centroid": centroid,
                            "reason": "sin casco",
                            "confidence": helmet_result.confidence,
                            "audit_image_path": crop_path,
                            "crop_rgb": crop_rgb,
                        }
                    )

            people_results.append(person_result)

        return {
            "frame_idx": frame_idx,
            "num_persons": len(people_results),
            "num_violations": len(violations),
            "required_zone_polygon": self.required_zone_polygon,
            "persons": people_results,
            "violations": violations,
        }

    def draw_overlays(self, frame_bgr: np.ndarray, results: Dict) -> np.ndarray:
        """Dibuja zona ROI y estado por persona sobre el frame."""
        out = frame_bgr.copy()

        if len(self.required_zone_polygon) >= 3:
            poly_np = np.array(self.required_zone_polygon, dtype=np.int32)
            cv2.polylines(out, [poly_np], isClosed=True, color=(255, 215, 0), thickness=2)
            cv2.putText(
                out,
                "Zona Requerida",
                tuple(poly_np[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 215, 0),
                2,
            )

        for person in results["persons"]:
            x1, y1, x2, y2 = person["bbox_pixels"]
            cx, cy = person["centroid"]
            in_zone = person["in_required_zone"]
            helmet_result = person["helmet_result"]

            if not in_zone:
                color = (120, 120, 120)
                status = "fuera zona"
            elif helmet_result and helmet_result["is_compliant"]:
                color = (0, 200, 0)
                status = f"con casco ({helmet_result['confidence']:.2f})"
            else:
                color = (0, 0, 255)
                conf = helmet_result["confidence"] if helmet_result else 0.0
                status = f"sin casco ({conf:.2f})"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out, (cx, cy), 4, color, -1)
            cv2.putText(
                out,
                f"P{person['person_id']} {status}",
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.putText(
            out,
            f"Personas:{results['num_persons']} Violaciones:{results['num_violations']}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return out

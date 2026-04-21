"""Detectores EPP extensibles basados en ViT.

Este modulo define una jerarquia abierta/cerrada para detectores de EPP:
- EPPDetector (base abstracta)
- HelmetDetector (implementacion concreta)

Se pueden agregar detectores futuros (GloveDetector, VestDetector) heredando
de EPPDetector sin modificar el flujo principal del sistema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import EPP_CLASS_NAMES, EPP_HELMET_LABEL
from models.vit_epp_classifier import ViTEPPClassifier


@dataclass
class EPPDetectionResult:
    """Resultado estandar de cualquier detector EPP."""

    detector_name: str
    label: str
    confidence: float
    is_compliant: bool


class EPPDetector(ABC):
    """Interfaz base para detectores EPP.

    Principio abierto/cerrado:
    - Cerrado a modificacion del flujo principal.
    - Abierto a extension mediante nuevas subclases.
    """

    def __init__(self, detector_name: str, device: str = "directml", threshold: float = 0.5) -> None:
        self.detector_name = detector_name
        self.device = device
        self.threshold = threshold

    @abstractmethod
    def detect(self, crop_rgb: np.ndarray) -> EPPDetectionResult:
        """Clasifica cumplimiento EPP para un crop de persona en RGB."""


class HelmetDetector(EPPDetector):
    """Detector de casco usando ViT multi-label.

    Interpreta la salida de la clase configurada para casco y estandariza
    el resultado en formato EPPDetectionResult.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "google/vit-base-patch16-224",
        device: str = "directml",
        threshold: float = 0.5,
        helmet_label: str = EPP_HELMET_LABEL,
        class_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__(detector_name="helmet", device=device, threshold=threshold)

        self.helmet_label = helmet_label
        self.class_names = list(class_names) if class_names is not None else list(EPP_CLASS_NAMES)

        self.model = self._load_model(model_path=model_path, model_name=model_name)
        self.processor = self.model.get_processor()

    def _load_model(self, model_path: Optional[str], model_name: str) -> ViTEPPClassifier:
        if model_path and Path(model_path).exists():
            model = ViTEPPClassifier.load_model(
                model_path,
                device=self.device,
                class_names=self.class_names,
            )
        else:
            model = ViTEPPClassifier(model_name=model_name, class_names=self.class_names)
            model.to(self.device)
            model.eval()
        return model

    def detect(self, crop_rgb: np.ndarray) -> EPPDetectionResult:
        if crop_rgb is None or crop_rgb.size == 0:
            return EPPDetectionResult(
                detector_name=self.detector_name,
                label="sin casco",
                confidence=0.0,
                is_compliant=False,
            )

        processed = self.processor(images=crop_rgb, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)
        pred = self.model.predict(pixel_values, threshold=self.threshold)

        helmet_key = self.helmet_label
        if helmet_key not in pred["classes"]:
            if len(pred["classes"]) == 1:
                helmet_key = next(iter(pred["classes"]))
            else:
                raise KeyError(f"Etiqueta de casco no encontrada: {self.helmet_label}")

        has_helmet = bool(pred["classes"][helmet_key])
        helmet_conf = float(pred["probabilities"][helmet_key])

        return EPPDetectionResult(
            detector_name=self.detector_name,
            label="con casco" if has_helmet else "sin casco",
            confidence=helmet_conf,
            is_compliant=has_helmet,
        )

"""Detectores EPP extensibles basados en ViT"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

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

    def __init__(self, detector_name: str, device: str = "directml", threshold: float = 0.5) -> None:
        self.detector_name = detector_name
        self.device = device
        self.threshold = threshold

    @abstractmethod
    def detect(self, crop_rgb: np.ndarray) -> EPPDetectionResult:
        """Clasifica cumplimiento EPP para un crop de persona en RGB."""


class EPPClassifierDetector(EPPDetector):
    """Detector EPP generico usando ViT multi-label.

    Configurable para cualquier tipo de EPP: casco, chaleco, guantes, etc.
    Para multi-EPP: instanciar uno por clase o entrenar ViT con N labels.
    """

    def __init__(
        self,
        epp_label: str,
        model_path: Optional[str] = None,
        model_name: str = "google/vit-base-patch16-224",
        device: str = "directml",
        threshold: float = 0.5,
        class_names: Optional[list[str]] = None,
        present_text: Optional[str] = None,
        absent_text: Optional[str] = None,
    ) -> None:
        super().__init__(detector_name=epp_label, device=device, threshold=threshold)

        self.epp_label = epp_label
        self.class_names = list(class_names) if class_names is not None else list(EPP_CLASS_NAMES)
        self.present_text = present_text or f"con {epp_label}"
        self.absent_text = absent_text or f"sin {epp_label}"

        self.model = self._load_model(model_path=model_path, model_name=model_name)
        self.processor = self.model.get_processor()

    def _load_model(self, model_path: Optional[str], model_name: str) -> ViTEPPClassifier:
        if model_path and Path(model_path).exists():
            return ViTEPPClassifier.load_model(
                model_path,
                device=self.device,
                class_names=self.class_names,
            )
        model = ViTEPPClassifier(model_name=model_name, class_names=self.class_names)
        model.to(self.device)
        model.eval()
        return model

    def detect(self, crop_rgb: np.ndarray) -> EPPDetectionResult:
        if crop_rgb is None or crop_rgb.size == 0:
            return EPPDetectionResult(
                detector_name=self.detector_name,
                label=self.absent_text,
                confidence=0.0,
                is_compliant=False,
            )

        processed = self.processor(images=crop_rgb, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)
        pred = self.model.predict(pixel_values, threshold=self.threshold)

        epp_key = self.epp_label
        if epp_key not in pred["classes"]:
            if len(pred["classes"]) == 1:
                epp_key = next(iter(pred["classes"]))
            else:
                raise KeyError(f"Etiqueta EPP no encontrada: {self.epp_label}")

        is_present = bool(pred["classes"][epp_key])
        confidence = float(pred["probabilities"][epp_key])

        return EPPDetectionResult(
            detector_name=self.detector_name,
            label=self.present_text if is_present else self.absent_text,
            confidence=confidence,
            is_compliant=is_present,
        )


class HelmetDetector(EPPClassifierDetector):
    """Detector de casco. API backward-compatible con la version anterior."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "google/vit-base-patch16-224",
        device: str = "directml",
        threshold: float = 0.5,
        helmet_label: str = EPP_HELMET_LABEL,
        class_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            epp_label=helmet_label,
            model_path=model_path,
            model_name=model_name,
            device=device,
            threshold=threshold,
            class_names=class_names,
        )

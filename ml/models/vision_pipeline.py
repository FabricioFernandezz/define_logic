"""
Pipeline completo de vision por computadora:
1. Deteccion de personas con YOLO
2. Clasificacion de EPP por persona con ViT
"""
import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from config import EPP_CLASS_NAMES
from models.person_detector import PersonDetector
from models.vit_epp_classifier import ViTEPPClassifier


class EPPVisionPipeline:
    """
    Pipeline E2E de deteccion y clasificacion de EPP
    
    Flujo:
    1. Carga imagen
    2. YOLO detecta personas → bounding boxes
    3. Por cada persona: crop + preprocessing
    4. ViT clasifica EPP configurables
    5. Retorna resultados estructurados
    """
    
    def __init__(self, 
                 yolo_model='yolov8n.pt',
                 vit_model_path=None,
                 device='directml',
                 yolo_conf=0.5,
                 vit_threshold=0.5,
                 crop_padding=10):
        """
        Inicializa el pipeline
        
        Args:
            yolo_model: Nombre del modelo YOLO
            vit_model_path: Ruta al modelo ViT guardado (si None, usa ViT pre-entrenado)
            device: 'directml', 'cpu'
            yolo_conf: Umbral confianza YOLO
            vit_threshold: Umbral para clasificación ViT (0-1)
            crop_padding: Píxeles de padding alrededor de personas
        """
        self.device = device
        self.yolo_conf = yolo_conf
        self.vit_threshold = vit_threshold
        self.crop_padding = crop_padding
        self.class_names = list(EPP_CLASS_NAMES)
        
        # Inicializar modelos
        print("[INIT] Cargando detector de personas...")
        self.yolo = PersonDetector(
            model_name=yolo_model,
            device=device,
            conf_threshold=yolo_conf
        )
        
        print("[INIT] Cargando ViT classifier...")
        if vit_model_path and Path(vit_model_path).exists():
            self.vit = ViTEPPClassifier.load_model(
                vit_model_path,
                device=device,
                class_names=self.class_names,
            )
        else:
            self.vit = ViTEPPClassifier(class_names=self.class_names).to(device)
            self.vit.eval()
        
        self.vit_processor = self.vit.get_processor()
        print("[OK] Pipeline inicializado")

    def _run_inference_on_rgb(self, image_rgb: np.ndarray, image_ref: str, return_crops: bool = False) -> Dict:
        """Ejecuta deteccion+clasificacion sobre una imagen RGB en memoria."""
        detections = self.yolo.detect(image_rgb, conf=self.yolo_conf)

        results = {
            'image_path': str(image_ref),
            'image_shape': image_rgb.shape,
            'num_persons': len(detections),
            'persons': []
        }

        for person_id, detection in enumerate(detections):
            bbox_pixels = detection['bbox_pixels']
            crop = self.yolo.crop_person(image_rgb, bbox_pixels, padding=self.crop_padding)
            vit_input = self._prepare_vit_input(crop)
            epp_pred = self.vit.predict(vit_input, threshold=self.vit_threshold)

            epp = {}
            for class_name in self.class_names:
                epp[class_name] = {
                    'present': bool(epp_pred['classes'].get(class_name, False)),
                    'confidence': float(epp_pred['probabilities'].get(class_name, 0.0)),
                }

            person_result = {
                'person_id': person_id,
                'bbox_pixels': bbox_pixels,
                'bbox_norm': detection['bbox_norm'],
                'detection_conf': detection['confidence'],
                'epp': epp,
            }

            if return_crops:
                person_result['crop'] = crop

            results['persons'].append(person_result)

        return results
    
    def process_image(self, image_path: str, return_crops=False) -> Dict:
        """
        Procesa una imagen completa del inicio al fin
        
        Args:
            image_path: Ruta a la imagen
            return_crops: Si True, incluye crops en resultados
        
        Returns:
            {
                'image_path': str,
                'image_shape': (H, W, C),
                'num_persons': int,
                'persons': [
                    {
                        'person_id': int,
                        'bbox_pixels': (x1, y1, x2, y2),
                        'bbox_norm': (x1_norm, y1_norm, x2_norm, y2_norm),
                        'detection_conf': float,
                        'epp': { 'label': {'present': bool, 'confidence': float} },
                        'crop': np.ndarray (opcional)
                    },
                    ...
                ]
            }
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar imagen: {image_path}")
        
        # Convertir BGR -> RGB para YOLO/ViT
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._run_inference_on_rgb(image_rgb, image_path, return_crops=return_crops)

    def process_frame(self, frame: np.ndarray, return_crops: bool = False, input_is_bgr: bool = True) -> Dict:
        """
        Procesa un frame en memoria (sin escribir a disco).

        Args:
            frame: Frame en memoria.
            return_crops: Si True, incluye crops por persona.
            input_is_bgr: True para frames OpenCV; False si ya viene en RGB.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame vacio o invalido")

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if input_is_bgr else frame
        return self._run_inference_on_rgb(image_rgb, "<frame>", return_crops=return_crops)
    
    def _prepare_vit_input(self, crop_rgb: np.ndarray) -> torch.Tensor:
        """
        Prepara un crop para entrada a ViT
        
        Args:
            crop_rgb: Imagen recortada en RGB
        
        Returns:
            Tensor procesado (1, 3, 224, 224)
        """
        # Procesar con el preprocesador de ViT
        processed = self.vit_processor(
            images=crop_rgb,
            return_tensors='pt'
        )
        
        # Mover a device
        pixel_values = processed['pixel_values'].to(self.device)
        
        return pixel_values
    
    def visualize_results(self, image_path: str, results: Dict, 
                         output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualiza detecciones y clasificaciones sobre la imagen
        
        Args:
            image_path: Ruta a la imagen original
            results: Resultados del procesamiento
            output_path: Ruta donde guardar (opcional)
        
        Returns:
            Imagen anotada
        """
        import cv2
        
        # Cargar imagen
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Colores
        color_person = (0, 255, 0)  # Verde para persona
        color_with_epp = (0, 255, 0)  # Verde si tiene EPP
        color_without_epp = (0, 0, 255)  # Rojo si falta EPP
        color_text = (255, 255, 255)  # Blanco
        
        # Anotar cada persona
        for person in results['persons']:
            x1, y1, x2, y2 = person['bbox_pixels']
            det_conf = person['detection_conf']
            epp = person['epp']
            
            # Determinar color según EPP compliance
            has_all_epp = all(
                epp.get(item, {}).get('present', False) for item in self.class_names
            )
            bbox_color = color_with_epp if has_all_epp else color_without_epp
            
            # Dibujar bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), bbox_color, 2)
            
            # Etiqueta EPP
            epp_status = []
            for epp_name in self.class_names:
                present = epp.get(epp_name, {}).get('present', False)
                conf = epp.get(epp_name, {}).get('confidence', 0.0)
                status = "✓" if present else "✗"
                epp_status.append(f"{epp_name}{status} ({conf:.2f})")
            
            # Texto
            text = f"Persona {person['person_id']} | Det: {det_conf:.2f}"
            cv2.putText(image_rgb, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color_text, 2)
            
            for idx, status_txt in enumerate(epp_status):
                cv2.putText(image_rgb, status_txt, (x1, y1 - 10 + idx * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)
        
        # Convertir a BGR para cv2
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Guardar si se especifica
        if output_path:
            cv2.imwrite(output_path, image_bgr)
            print(f"[OK] Imagen anotada guardada en: {output_path}")
        
        return image_bgr
    
    def print_results(self, results: Dict):
        """Imprime resultados de forma legible"""
        print("\n" + "="*70)
        print(f"RESULTADOS - {results['image_path']}")
        print(f"Imagen: {results['image_shape']}")
        print(f"Personas detectadas: {results['num_persons']}")
        print("="*70)
        
        for person in results['persons']:
            print(f"\nPersona {person['person_id']}:")
            print(f"  Bbox (píxeles): {person['bbox_pixels']}")
            print(f"  Confianza detección: {person['detection_conf']:.4f}")
            print(f"  EPP:")
            for epp_name, epp_data in person['epp'].items():
                status = "✓ PRESENTE" if epp_data['present'] else "✗ AUSENTE"
                print(f"    {epp_name.upper()}: {status} (conf: {epp_data['confidence']:.4f})")


if __name__ == "__main__":
    """Ejemplo de uso del pipeline"""
    import sys
    
    print("=== EPP Vision Pipeline ===\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Inicializar pipeline
    pipeline = EPPVisionPipeline(
        yolo_model='yolov8n.pt',
        vit_model_path=None,  # Usar ViT pre-entrenado
        device=device,
        yolo_conf=0.5,
        vit_threshold=0.5
    )
    
    print("\n[OK] Pipeline listo para procesar imágenes")
    print("Uso: python vision_pipeline.py <ruta_imagen> [ruta_salida_anotada]")

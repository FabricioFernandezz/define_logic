import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import os


class PersonDetector:
    def __init__(self, model_name='yolov8n.pt', device='directml', conf_threshold=0.5):
        self.model = YOLO(model_name)
        self.device = device
        self.conf_threshold = conf_threshold
        self.model.to(device)
        self.person_class_id = 0
        
    def detect(self, image: np.ndarray, conf=None) -> List[Dict]:
        if conf is None:
            conf = self.conf_threshold
        
        results = self.model(image, conf=conf, verbose=False)
        
        detections = []
        img_h, img_w = image.shape[:2]
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                
                if cls_id != self.person_class_id:
                    continue
                
                conf_score = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                x1_norm = x1 / img_w
                y1_norm = y1 / img_h
                x2_norm = x2 / img_w
                y2_norm = y2 / img_h
                
                detections.append({
                    'bbox_pixels': (x1, y1, x2, y2),
                    'bbox_norm': (x1_norm, y1_norm, x2_norm, y2_norm),
                    'confidence': conf_score,
                    'class_id': cls_id,
                    'class_name': 'persona'
                })
        
        return detections
    
    def crop_person(self, image: np.ndarray, bbox_pixels: Tuple[int, int, int, int], 
                    padding=0) -> np.ndarray:
        x1, y1, x2, y2 = bbox_pixels
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        crop = image[y1:y2, x1:x2]
        return crop
    
    def get_model_info(self):
    
        return {
            'model_name': self.model.model_name,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'num_parameters': sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, 'parameters') else 'N/A'
        }


if __name__ == "__main__":
    print("=== Test PersonDetector ===")
    
    device = 'directml'
    detector = PersonDetector(device=device, conf_threshold=0.5)
    
    print(f"[OK] Detector YOLO inicializado")
    print(f"   Info: {detector.get_model_info()}")

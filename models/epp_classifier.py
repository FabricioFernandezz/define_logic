"""
EPP (Equipo de Protección Personal) Classifier
Detecta cascos, guantes y chalecos de seguridad
"""
import os
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime


class EPPClassifier:
    """Clasificador de equipos de protección personal usando YOLOv8"""
    
    def __init__(self, model_path=None):
        """
        Inicializa el clasificador EPP
        
        Args:
            model_path: Ruta al modelo entrenado. Si es None, usa modelo base
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Inicializa con modelo base para entrenamiento
            self.model = YOLO('yolov8n.pt')
        
        # Mapeo de clases
        self.class_names = {
            0: 'casco',
            1: 'guantes', 
            2: 'chaleco'
        }
    
    def train(self, 
              data_yaml_path,
              epochs=100,
              imgsz=640,
              batch=16,
              patience=50,
              save_dir='runs/train',
              device=''):
        """
        Entrena el modelo con buenas prácticas
        
        Args:
            data_yaml_path: Ruta al archivo YAML con configuración del dataset
            epochs: Número de épocas de entrenamiento
            imgsz: Tamaño de imagen para entrenamiento
            batch: Tamaño del batch
            patience: Paciencia para early stopping
            save_dir: Directorio para guardar resultados
            device: Device para entrenamiento ('', 'cpu', '0', '0,1', etc.)
        
        Returns:
            Resultados del entrenamiento
        """
        # Verificar que existe el archivo de configuración
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"No se encontró el archivo de datos: {data_yaml_path}")
        
        # Configuración de entrenamiento con buenas prácticas
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            
            # Optimización
            optimizer='Adam',  # Adam suele funcionar bien
            lr0=0.001,  # Learning rate inicial
            lrf=0.01,   # Learning rate final (lr0 * lrf)
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Augmentación de datos
            hsv_h=0.015,  # Cambio de matiz
            hsv_s=0.7,    # Cambio de saturación
            hsv_v=0.4,    # Cambio de valor
            degrees=0.0,  # Rotación
            translate=0.1,  # Traslación
            scale=0.5,    # Escala
            shear=0.0,    # Corte
            perspective=0.0,  # Perspectiva
            flipud=0.0,   # Volteo vertical
            fliplr=0.5,   # Volteo horizontal
            mosaic=1.0,   # Mosaic augmentation
            mixup=0.0,    # Mixup augmentation
            copy_paste=0.0,  # Copy-paste augmentation
            
            # Early stopping y validación
            patience=patience,
            save=True,
            save_period=-1,  # Guarda checkpoint cada N épocas (-1 = solo mejor)
            val=True,
            plots=True,
            
            # Hardware
            device=device,
            workers=8,
            
            # Otros
            project=save_dir,
            name=f'epp_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=True,
            pretrained=True,
            verbose=True,
            seed=42,  # Reproducibilidad
            deterministic=True,
        )
        
        print(f"\n✓ Entrenamiento completado")
        print(f"✓ Modelo guardado en: {results.save_dir}")
        
        return results
    
    def validate(self, data_yaml_path, split='val'):
        """
        Valida el modelo en el conjunto de validación o test
        
        Args:
            data_yaml_path: Ruta al archivo YAML con configuración del dataset
            split: 'val' o 'test'
        
        Returns:
            Métricas de validación
        """
        metrics = self.model.val(
            data=data_yaml_path,
            split=split,
            save_json=True,
            plots=True
        )
        
        print(f"\n=== Métricas de validación ===")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def detect(self, frame, conf_threshold=0.5):
        """
        Detecta EPP en un frame
        
        Args:
            frame: Imagen/frame para detección
            conf_threshold: Umbral de confianza mínimo
        
        Returns:
            Lista de detecciones con formato:
            [{
                'class': nombre_clase,
                'class_id': id_clase,
                'bbox': (x1, y1, x2, y2),
                'confidence': confianza
            }]
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    'class': self.class_names.get(cls_id, f'class_{cls_id}'),
                    'class_id': cls_id,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
        
        return detections
    
    def export_model(self, format='onnx', output_path=None):
        """
        Exporta el modelo a diferentes formatos
        
        Args:
            format: Formato de exportación ('onnx', 'torchscript', 'tflite', etc.)
            output_path: Ruta donde guardar el modelo exportado
        
        Returns:
            Ruta del modelo exportado
        """
        exported_path = self.model.export(format=format)
        print(f"✓ Modelo exportado a formato {format}: {exported_path}")
        return exported_path
    
    def load_model(self, model_path):
        """
        Carga un modelo previamente entrenado
        
        Args:
            model_path: Ruta al modelo .pt
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
        
        self.model = YOLO(model_path)
        print(f"✓ Modelo cargado desde: {model_path}")
    
    def get_model_info(self):
        """
        Obtiene información sobre el modelo
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            'type': type(self.model).__name__,
            'task': self.model.task,
            'names': self.class_names,
            'device': next(self.model.model.parameters()).device if hasattr(self.model, 'model') else 'unknown'
        }


def create_data_yaml(dataset_paths, output_path='data/epp_data.yaml'):
    """
    Crea el archivo YAML de configuración para el dataset
    
    Args:
        dataset_paths: Lista de rutas a los datasets
        output_path: Ruta donde guardar el archivo YAML
    
    Example YAML structure:
        path: /path/to/dataset
        train: images/train
        val: images/val
        test: images/test  # opcional
        
        names:
          0: casco
          1: guantes
          2: chaleco
    """
    import yaml
    
    # Template básico
    yaml_content = {
        'path': dataset_paths[0] if dataset_paths else 'data/epp_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'casco',
            1: 'guantes',
            2: 'chaleco'
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ Archivo de configuración creado: {output_path}")
    return output_path


if __name__ == "__main__":
    """Ejemplo de uso del clasificador EPP"""
    
    # Ejemplo 1: Crear archivo de configuración de datos
    print("=== Ejemplo 1: Crear configuración de datos ===")
    from config import EPP_DATASET_PATHS
    yaml_path = create_data_yaml(EPP_DATASET_PATHS)
    
    # Ejemplo 2: Entrenar modelo
    print("\n=== Ejemplo 2: Entrenar modelo ===")
    classifier = EPPClassifier()
    
    # Entrenar (descomenta para ejecutar)
    # results = classifier.train(
    #     data_yaml_path=yaml_path,
    #     epochs=100,
    #     imgsz=640,
    #     batch=16,
    #     device=''  # '' para auto-detectar GPU/CPU
    # )
    
    # Ejemplo 3: Validar modelo
    print("\n=== Ejemplo 3: Validar modelo ===")
    # metrics = classifier.validate(yaml_path)
    
    # Ejemplo 4: Detectar en una imagen
    print("\n=== Ejemplo 4: Detectar EPP ===")
    # import cv2
    # frame = cv2.imread('test_image.jpg')
    # detections = classifier.detect(frame, conf_threshold=0.5)
    # for det in detections:
    #     print(f"Detectado {det['class']} con confianza {det['confidence']:.2f}")
    
    # Ejemplo 5: Exportar modelo
    print("\n=== Ejemplo 5: Exportar modelo ===")
    # classifier.export_model(format='onnx')
    
    print("\n✓ Módulo EPP Classifier listo para usar")

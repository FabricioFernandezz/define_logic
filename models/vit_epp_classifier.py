"""
Vision Transformer (ViT) para clasificación multi-label de EPP
Casco, Guantes, Chaleco
"""
import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel
from typing import Dict, List, Tuple
import os
from utils.device_manager import get_device


class ViTEPPClassifier(nn.Module):
    """
    Clasificador ViT para atributos EPP multi-label.
    
    Arquitectura:
    - ViT base pre-entrenado (google/vit-base-patch16-224)
    - Capa de clasificación adaptada para 3 salidas (casco, guantes, chaleco)
    - Sigmoid para multi-label + BCEWithLogitsLoss
    """
    
    def __init__(self, model_name="google/vit-base-patch16-224", num_labels=3, dropout=0.2):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.class_names = ['casco', 'guantes', 'chaleco']
        
        # Cargar ViT pre-entrenado
        self.vit = ViTModel.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Obtener dimensión de salida de ViT
        vit_hidden_size = self.vit.config.hidden_size
        
        # Cabeza de clasificación multi-label
        self.classifier_head = nn.Sequential(
            nn.Linear(vit_hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)  # sin activación, usaremos BCEWithLogitsLoss
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, pixel_values, apply_sigmoid=False):
        """
        Forward pass
        
        Args:
            pixel_values: Imagen procesada por ViTImageProcessor
            apply_sigmoid: Si True, aplica sigmoid (para inferencia)
        
        Returns:
            logits: Tensor de shape (batch_size, num_labels) sin activación
            probs: Tensor de shape (batch_size, num_labels) con sigmoid aplicado
        """
        # ViT forward
        outputs = self.vit(pixel_values)
        
        # Usar el [CLS] token (índice 0)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # Cabeza de clasificación
        logits = self.classifier_head(cls_token)
        
        if apply_sigmoid:
            probs = self.sigmoid(logits)
            return logits, probs
        
        return logits
    
    def predict(self, pixel_values, threshold=0.5):
        """
        Predicción con etiquetas binarias
        
        Args:
            pixel_values: Imagen procesada
            threshold: Umbral para binariación (default 0.5)
        
        Returns:
            predictions: Dict con clases y confianzas
        """
        with torch.no_grad():
            logits, probs = self.forward(pixel_values, apply_sigmoid=True)
        
        # Binarizar
        predictions_binary = (probs > threshold).int()
        
        # Formato legible
        batch_results = []
        for i in range(probs.shape[0]):
            result = {
                'classes': {},
                'probabilities': {},
                'binary': {}
            }
            for j, class_name in enumerate(self.class_names):
                result['classes'][class_name] = bool(predictions_binary[i, j].item())
                result['probabilities'][class_name] = float(probs[i, j].item())
                result['binary'][class_name] = int(predictions_binary[i, j].item())
            batch_results.append(result)
        
        return batch_results if len(batch_results) > 1 else batch_results[0]
    
    def get_processor(self):
        """Retorna el procesador de imágenes"""
        return self.processor
    
    def save_model(self, save_path):
        """Guarda el modelo entrenado"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"[OK] Modelo guardado en: {save_path}")
    
    @staticmethod
    def load_model(model_path, device='directml'):
        """Carga un modelo previamente guardado"""
        model = ViTEPPClassifier()
        # DirectML no necesita map_location específico
        model.load_state_dict(torch.load(model_path))
        model.to(get_device(device))
        model.eval()
        print(f"[OK] Modelo cargado desde: {model_path}")
        return model


if __name__ == "__main__":
    """Test básico del modelo ViT"""
    print("=== Test ViTEPPClassifier ===")
    
    # Inicializar modelo
    device = 'directml'
    print(f"Device: {device}")
    
    model = ViTEPPClassifier().to(device)
    processor = model.get_processor()
    
    print(f"[OK] Modelo ViT cargado")
    print(f"   Clases: {model.class_names}")
    print(f"   Número de parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Crear dummy input
    print("\n[TEST] Forward pass con tensor dummy...")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        logits, probs = model(dummy_input, apply_sigmoid=True)
    
    print(f"   Logits shape: {logits.shape}")
    print(f"   Probs shape: {probs.shape}")
    print(f"   Probs: {probs}")
    
    # Test predicción
    print("\n[TEST] Predicción...")
    predictions = model.predict(dummy_input, threshold=0.5)
    for i, pred in enumerate(predictions):
        print(f"   Persona {i+1}: {pred['classes']}")
        print(f"      Confianzas: {pred['probabilities']}")

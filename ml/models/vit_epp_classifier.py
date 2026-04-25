import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel
from typing import Dict, List, Tuple
import os
import sys
import types
from utils.device_manager import get_device


DEFAULT_CLASS_NAMES = ["casco"]


class ViTEPPClassifier(nn.Module):
    def __init__(
        self,
        model_name="google/vit-base-patch16-224",
        num_labels=None,
        class_names=None,
        dropout=0.2,
    ):
        super().__init__()

        if class_names is None:
            class_names = list(DEFAULT_CLASS_NAMES)
        else:
            class_names = list(class_names)

        if num_labels is None:
            num_labels = len(class_names)
        elif num_labels != len(class_names):
            raise ValueError(
                "num_labels debe coincidir con len(class_names). "
                f"num_labels={num_labels}, class_names={class_names}"
            )

        self.model_name = model_name
        self.num_labels = num_labels
        self.class_names = class_names
        
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

        outputs = self.vit(pixel_values)
        
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier_head(cls_token)
        
        if apply_sigmoid:
            probs = self.sigmoid(logits)
            return logits, probs
        
        return logits
    
    def predict(self, pixel_values, threshold=0.5):

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
        return self.processor
    
    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"[OK] Modelo guardado en: {save_path}")
    
    @staticmethod
    def load_model(
        model_path,
        device='directml',
        class_names=None,
        num_labels=None,
        model_name="google/vit-base-patch16-224",
        dropout=0.2,
    ):
        model = ViTEPPClassifier(
            model_name=model_name,
            num_labels=num_labels,
            class_names=class_names,
            dropout=dropout,
        )

        def _torch_load_cpu(path):
            # Cargar en CPU evita depender del backend original del checkpoint.
            try:
                return torch.load(path, map_location='cpu', weights_only=True)
            except ModuleNotFoundError as exc:
                if 'torch.privateuseone' not in str(exc):
                    raise
                # Compatibilidad con checkpoints antiguos guardados desde DirectML.
                sys.modules.setdefault('torch.privateuseone', types.ModuleType('torch.privateuseone'))
                try:
                    return torch.load(path, map_location='cpu', weights_only=True)
                except TypeError:
                    return torch.load(path, map_location='cpu')
                except Exception as inner_exc:
                    if 'Weights only load failed' in str(inner_exc):
                        return torch.load(path, map_location='cpu', weights_only=False)
                    raise
            except TypeError:
                return torch.load(path, map_location='cpu')
            except Exception as exc:
                # Checkpoints legacy pueden no soportar weights_only=True.
                if 'Weights only load failed' in str(exc):
                    return torch.load(path, map_location='cpu', weights_only=False)
                raise

        checkpoint = _torch_load_cpu(model_path)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict):
            # Quita prefijo comun cuando se guardo con DataParallel/DDP.
            state_dict = {
                (k[7:] if k.startswith('module.') else k): v
                for k, v in state_dict.items()
            }

        model_state = model.state_dict()
        filtered_state = {}
        mismatched_keys = []
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                filtered_state[key] = value
            else:
                mismatched_keys.append(key)

        load_report = model.load_state_dict(filtered_state, strict=False)
        if load_report.missing_keys or load_report.unexpected_keys:
            print("[WARN] Carga parcial del checkpoint ViT")
            if load_report.missing_keys:
                print(f"       Missing keys: {load_report.missing_keys[:5]}")
            if load_report.unexpected_keys:
                print(f"       Unexpected keys: {load_report.unexpected_keys[:5]}")
        if mismatched_keys:
            print("[WARN] Pesos omitidos por tamano incompatible")
            print(f"       Mismatched keys: {mismatched_keys[:5]}")

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

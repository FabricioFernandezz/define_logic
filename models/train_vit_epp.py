import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score
import json
from datetime import datetime
import sys
import warnings
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vit_epp_classifier import ViTEPPClassifier
from utils.device_manager import get_device, get_device_strict, print_device_info
from config import (
    DEVICE,
    ViT_TRAINING_CONFIG,
    ViT_DATASET_ROOT,
    ViT_MODEL_PATH,
    ViT_SIMPLE_MODEL_PATH,
)

class ViTEPPDataset(Dataset):
    
    def __init__(self, data_dir, split='train', processor=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.processor = processor
        self.class_names = ['casco', 'guantes', 'chaleco']
        self.metadata_keys = {'filename', 'source', 'orig_size', 'num_annotations', 'orig_bbox'}
        self._ignored_label_keys = set()
        self._ignored_label_warn_count = 0

        # Resolver alias de split: val <-> valid
        split_aliases = {
            'train': ['train'],
            'val': ['val', 'valid'],
            'valid': ['valid', 'val'],
            'test': ['test'],
        }
        candidate_splits = split_aliases.get(split, [split])

        self.split_dir = None
        for split_name in candidate_splits:
            candidate_dir = self.data_dir / split_name
            if candidate_dir.exists():
                self.split_dir = candidate_dir
                break

        if self.split_dir is None:
            searched = ', '.join(str(self.data_dir / s) for s in candidate_splits)
            raise FileNotFoundError(f"No split directory found for '{split}'. Searched: {searched}")

        # Buscar imágenes con extensiones comunes
        image_patterns = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        self.image_files = sorted(
            p for pattern in image_patterns for p in self.split_dir.glob(pattern)
        )

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No images found in {self.split_dir}")

        print(f"[INFO] Dataset {split} -> {self.split_dir.name}: {len(self.image_files)} muestras")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):

        from PIL import Image
        
        img_path = self.image_files[idx]
        label_path = self.data_dir / 'labels' / (img_path.stem + '.json')
        
        # Cargar imagen
        image = Image.open(img_path).convert('RGB')
        
        # Cargar etiquetas
        with open(label_path, 'r') as f:
            label_dict = json.load(f)

        if not isinstance(label_dict, dict):
            raise ValueError(f"Formato de label invalido en {label_path}")
        # Ignora labels fuera de casco/guantes/chaleco (sin contar metadatos conocidos).
        extra_keys = set(label_dict.keys()) - set(self.class_names) - self.metadata_keys
        if extra_keys:
            self._ignored_label_keys.update(extra_keys)
            if self._ignored_label_warn_count < 3:
                print(
                    f"[WARN] Labels no objetivo ignorados en {label_path.name}: "
                    f"{', '.join(sorted(extra_keys))}"
                )
                self._ignored_label_warn_count += 1
        
        # Convertir a tensor
        labels = torch.tensor([
            int(label_dict.get(cls, 0)) for cls in self.class_names
        ], dtype=torch.float32)
        
        # Procesar imagen
        if self.processor:
            inputs = self.processor(image, return_tensors='pt')
            pixel_values = inputs['pixel_values'].squeeze(0)
        else:
            pixel_values = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        # DirectML es sensible a tensores no contiguos en algunas rutas.
        pixel_values = pixel_values.float().contiguous()
        labels = labels.float().contiguous()

        return pixel_values, labels


class ViTEPPTrainer:
    
    def __init__(
        self,
        model,
        device='cpu',
        lr=1e-4,
        num_epochs=20,
        strict_no_cpu_fallback=True,
        patience=5,
        freeze_backbone_epochs=0,
        max_train_batches_per_epoch=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.strict_no_cpu_fallback = strict_no_cpu_fallback
        self.patience = patience
        self.freeze_backbone_epochs = max(0, int(freeze_backbone_epochs))
        self.max_train_batches_per_epoch = max_train_batches_per_epoch
        self._backbone_frozen = False

        if self.strict_no_cpu_fallback:
            # Convierte cualquier warning de fallback DML->CPU en error fatal.
            warnings.filterwarnings(
                "error",
                message=r".*fall back to run on the CPU.*",
                category=UserWarning,
            )
            # Ruta estricta DirectML: evita Adam/BCEWithLogits (ops con fallback conocido).
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            self.loss_fn = nn.MSELoss()
            print("[STRICT] Modo sin fallback CPU activo: optimizer=SGD, loss=MSE(sigmoid(logits))")
        else:
            # Ruta estándar (puede tener fallback en DirectML para algunas ops).
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': []
        }
        self._device_debug_printed = False

    def _set_backbone_trainable(self, trainable: bool):
        if not hasattr(self.model, 'vit'):
            return
        for p in self.model.vit.parameters():
            p.requires_grad = trainable
        self._backbone_frozen = not trainable
        state = 'congelado' if self._backbone_frozen else 'descongelado'
        print(f"[FAST] Backbone ViT {state}")

    def _compute_loss(self, logits, labels):
        if self.strict_no_cpu_fallback:
            return self.loss_fn(torch.sigmoid(logits), labels)
        return self.loss_fn(logits, labels)

    def _move_batch_to_device(self, inputs, labels):
        
        try:
            inputs = inputs.contiguous().to(self.device)
            labels = labels.contiguous().to(self.device)
            return inputs, labels
        except RuntimeError as e:
            msg = str(e).lower()
            if 'unknown error' in msg or 'dml' in msg:
                raise RuntimeError(
                    "\n"
                    + "=" * 70
                    + "\n[FATAL ERROR] Falla DirectML al mover batch a GPU\n"
                    + "Posible timeout/reinicio del driver AMD (TDR).\n"
                    + "Acciones recomendadas:\n"
                    + "  1. Reducir batch_size (recomendado: 4 o 2).\n"
                    + "  2. Cerrar apps que usen GPU (navegador/video/juegos).\n"
                    + "  3. Actualizar driver AMD (Adrenalin WHQL).\n"
                    + "  4. Reintentar entrenamiento tras reiniciar la GPU/PC.\n"
                    + "=" * 70
                ) from e
            raise
    
    def train_epoch(self, train_loader):
        
        self.model.train()
        total_loss = 0

        total_batches = len(train_loader)
        if self.max_train_batches_per_epoch is not None:
            total_batches = min(total_batches, int(self.max_train_batches_per_epoch))

        t0 = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx >= total_batches:
                break

            inputs, labels = self._move_batch_to_device(inputs, labels)

            if not self._device_debug_printed:
                model_device = next(self.model.parameters()).device
                print("[DEVICE] Verificacion de runtime:")
                print(f"  - Modelo en: {model_device}")
                print(f"  - Inputs en: {inputs.device}")
                print(f"  - Labels en: {labels.device}")
                
                # VALIDACIÓN ESTRICTA: FALLA si está en CPU
                if 'cpu' in str(model_device) or 'cpu' in str(inputs.device) or 'cpu' in str(labels.device):
                    raise RuntimeError(
                        f"\n{'='*70}\n"
                        f"[FATAL ERROR] ENTRENAMIENTO USANDO CPU EN VEZ DE GPU\n"
                        f"{'='*70}\n"
                        f"Modelo device: {model_device}\n"
                        f"Inputs device: {inputs.device}\n"
                        f"Labels device: {labels.device}\n"
                        f"\nEste proyecto requiere OBLIGATORIAMENTE GPU DirectML.\n"
                        f"No se permite calcular en CPU.\n"
                        f"\nVerifica que DirectML esté correctamente instalado:\n"
                        f"  pip install torch-directml\n"
                        f"{'='*70}\n"
                    )
                
                self._device_debug_printed = True
            
            # Forward
            logits = self.model(inputs, apply_sigmoid=False)
            loss = self._compute_loss(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % max(1, total_batches // 8) == 0:
                elapsed = time.time() - t0
                sec_per_batch = elapsed / (batch_idx + 1)
                eta_min = (sec_per_batch * (total_batches - (batch_idx + 1))) / 60.0
                print(
                    f"  Batch [{batch_idx+1}/{total_batches}], "
                    f"Loss: {loss.item():.4f}, "
                    f"{sec_per_batch:.2f}s/batch, ETA: {eta_min:.1f}m"
                )

        avg_loss = total_loss / total_batches
        return avg_loss
    
    def validate(self, val_loader):
        
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = self._move_batch_to_device(inputs, labels)
                
                logits = self.model(inputs, apply_sigmoid=False)
                loss = self._compute_loss(logits, labels)
                total_loss += loss.item()
                
                # Predicciones
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int().cpu().numpy()
                
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        
        return avg_loss, f1
    
    def fit(self, train_loader, val_loader, save_path=None):
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\n" + "="*70)
        print("ENTRENAMIENTO ViT EPP CLASSIFIER")
        print("="*70)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoca {epoch + 1}/{self.num_epochs}")

            # Warmup rápido: entrena primero la cabeza de clasificación.
            if self.freeze_backbone_epochs > 0 and epoch < self.freeze_backbone_epochs:
                if not self._backbone_frozen:
                    self._set_backbone_trainable(False)
            elif self._backbone_frozen:
                self._set_backbone_trainable(True)
            
            # Entrenamiento
            train_loss = self.train_epoch(train_loader)
            print(f"  Train Loss: {train_loss:.4f}")
            
            # Validación
            val_loss, val_f1 = self.validate(val_loader)
            print(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Guardar historia
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    self.model.save_model(save_path)
                    print(f"  [CHECKPOINT] Mejor modelo guardado")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n[EARLY STOPPING] Sin mejora en {self.patience} épocas")
                    break
        
        print("\n[OK] Entrenamiento completado")
        return self.history


def main():
    
    print("="*70)
    print("ENTRENAMIENTO ViT PARA CLASIFICACIÓN DE EPP")
    print("="*70)

    print("\n[INIT] Validando GPU obligatoria...")
    try:
        device = get_device_strict(force_gpu=True)
    except RuntimeError as e:
        print(str(e))
        print("\n[FATAL] ENTRENAMIENTO CANCELADO: GPU NO DISPONIBLE")
        sys.exit(1)
    
    dataset_root = ViT_DATASET_ROOT
    save_path = ViT_MODEL_PATH
    
    print("\n[INIT] Dispositivos disponibles:")
    print_device_info()
    print(f"\nDevice efectivo: {device}")
    print(f"Dataset: {dataset_root}")
    
    print("\n[INIT] Inicializando modelo ViT...")
    model = ViTEPPClassifier()

    if ViT_TRAINING_CONFIG.get('phase2_init_from_simple', True):
        phase1_path = Path(ViT_SIMPLE_MODEL_PATH)
        if phase1_path.exists():
            try:
                state_dict = torch.load(str(phase1_path), map_location='cpu')
                model.load_state_dict(state_dict, strict=True)
                print(f"[FAST] Pesos de fase 1 cargados desde: {phase1_path}")
            except Exception as e:
                print(f"[WARN] No se pudo cargar fase 1 ({phase1_path}): {e}")
                print("[WARN] Continuando desde ViT pre-entrenado base")
        else:
            print(f"[FAST] No existe fase 1 en {phase1_path}, usando base pre-entrenada")

    processor = model.get_processor()
    print("[OK] Modelo ViT inicializado")

    print("\n[DATA] Cargando datasets...")
    try:
        train_dataset = ViTEPPDataset(dataset_root, split='train', processor=processor)
        val_dataset = ViTEPPDataset(dataset_root, split='valid', processor=processor)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(f"\nEstructura esperada:")
        print(f"  {dataset_root}/")
        print(f"    train/")
        print(f"      person_0000.jpg")
        print(f"      person_0001.jpg")
        print(f"      ...")
        print(f"    valid/   (o val/)")
        print(f"      person_5000.jpg")
        print(f"      ...")
        print(f"    labels/")
        print(f"      person_0000.json  ({{\"casco\": 1, \"guantes\": 0, \"chaleco\": 1}})")
        print(f"      person_0001.json")
        print(f"      ...")
        return
    
    batch_size = ViT_TRAINING_CONFIG.get('batch_size', 32)
    if 'privateuseone' in str(device):
        # Tope conservador para estabilidad en DirectML (evita TDR/unknown error).
        directml_cap = ViT_TRAINING_CONFIG.get('directml_batch_cap', 4)
        if batch_size > directml_cap:
            print(f"[WARN] batch_size {batch_size} -> {directml_cap} para estabilidad DirectML")
            batch_size = directml_cap

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    strict_no_cpu_fallback = ViT_TRAINING_CONFIG.get('strict_no_cpu_fallback', True)
    freeze_backbone_epochs = ViT_TRAINING_CONFIG.get('freeze_backbone_epochs', 3)
    max_train_batches_per_epoch = ViT_TRAINING_CONFIG.get('max_train_batches_per_epoch', 800)
    patience = ViT_TRAINING_CONFIG.get('patience', 5)

    trainer = ViTEPPTrainer(
        model,
        device=device,
        lr=ViT_TRAINING_CONFIG.get('learning_rate', 1e-4),
        num_epochs=ViT_TRAINING_CONFIG.get('epochs', 20),
        strict_no_cpu_fallback=strict_no_cpu_fallback,
        patience=patience,
        freeze_backbone_epochs=freeze_backbone_epochs,
        max_train_batches_per_epoch=max_train_batches_per_epoch,
    )
    history = trainer.fit(train_loader, val_loader, save_path)
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO FINALIZADO")
    print("="*70)

    history_path = "runs/training/vit_history.json"
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n[OK] Modelo guardado en: {save_path}")
    print(f"[OK] Historia guardada en: {history_path}")


if __name__ == "__main__":
    main()

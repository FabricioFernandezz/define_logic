import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
    EPP_CLASS_NAMES,
    EPP_CSV_FILENAME,
    EPP_CSV_LABEL_MAP,
    ViT_TRAINING_CONFIG,
    ViT_DATASET_ROOT,
    ViT_MODEL_PATH,
)

class ViTEPPDataset(Dataset):
    
    def __init__(self, data_dir, split='train', processor=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.processor = processor
        self.class_names = list(EPP_CLASS_NAMES)
        self._class_name_lookup = {name.lower(): name for name in self.class_names}
        self._csv_label_map = {k.lower(): v for k, v in EPP_CSV_LABEL_MAP.items()}
        self._csv_class_indices = {}
        self._csv_raw_indices = {}
        self._csv_rows = {}
        self.label_source = "json"
        self._missing_csv_label_warned = False
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

        csv_path = self.split_dir / EPP_CSV_FILENAME
        if csv_path.exists():
            self._load_csv_labels(csv_path)
            self.label_source = "csv"
            self._filter_ambiguous_csv_rows()

        print(
            f"[INFO] Dataset {split} -> {self.split_dir.name}: {len(self.image_files)} muestras"
            f" ({self.label_source})"
        )

    def _load_csv_labels(self, csv_path: Path) -> None:
        with open(csv_path, newline='', encoding='utf-8') as file_handle:
            reader = csv.reader(file_handle)
            header = next(reader, None)
            if not header:
                raise ValueError(f"CSV sin encabezado: {csv_path}")

            normalized = [h.strip() for h in header]
            self._csv_raw_indices = {col_name.lower(): idx for idx, col_name in enumerate(normalized)}
            try:
                filename_idx = normalized.index("filename")
            except ValueError:
                filename_idx = 0

            for idx, col_name in enumerate(normalized):
                if idx == filename_idx:
                    continue
                col_norm = col_name.lower()
                mapped = self._csv_label_map.get(col_norm, col_name)
                mapped_norm = str(mapped).strip().lower()
                canonical = self._class_name_lookup.get(mapped_norm)
                if canonical:
                    self._csv_class_indices[canonical] = idx

            for row in reader:
                if not row or len(row) <= filename_idx:
                    continue
                filename = row[filename_idx].strip()
                if filename:
                    self._csv_rows[filename] = row

        if not self._csv_class_indices:
            raise ValueError(f"CSV sin columnas de clase conocidas: {csv_path}")

        missing = [name for name in self.class_names if name not in self._csv_class_indices]
        if missing:
            print(f"[WARN] CSV sin columnas para: {', '.join(missing)}")

    def _filter_ambiguous_csv_rows(self) -> None:
        """Filtra filas ambiguas para casco binario (Helmet y No Helmet simultaneos)."""
        if len(self.class_names) != 1:
            return

        only_class = self.class_names[0].strip().lower()
        if only_class != "casco":
            return

        helmet_idx = self._csv_raw_indices.get("helmet")
        no_helmet_idx = self._csv_raw_indices.get("no helmet")
        if no_helmet_idx is None:
            no_helmet_idx = self._csv_raw_indices.get("no_helmet")

        if helmet_idx is None or no_helmet_idx is None:
            return

        original_count = len(self.image_files)
        filtered_images = []
        dropped = 0

        for img_path in self.image_files:
            row = self._csv_rows.get(img_path.name)
            if row is None:
                continue
            try:
                helmet_val = int(float(row[helmet_idx])) if helmet_idx < len(row) else 0
                no_helmet_val = int(float(row[no_helmet_idx])) if no_helmet_idx < len(row) else 0
            except ValueError:
                helmet_val = 0
                no_helmet_val = 0

            # Si ambas clases están activas en una misma imagen, es ambiguo para
            # clasificación binaria por crop de persona.
            if helmet_val == 1 and no_helmet_val == 1:
                dropped += 1
                continue

            filtered_images.append(img_path)

        if filtered_images:
            self.image_files = filtered_images
        if dropped > 0:
            print(
                f"[INFO] Filtradas {dropped} muestras ambiguas Helmet/No Helmet "
                f"en {self.split_dir.name} (de {original_count})"
            )

    def _labels_from_csv(self, filename: str) -> torch.Tensor:
        row = self._csv_rows.get(filename)
        if row is None:
            if not self._missing_csv_label_warned:
                print(f"[WARN] CSV sin etiqueta para {filename}; usando ceros")
                self._missing_csv_label_warned = True
            values = [0 for _ in self.class_names]
        else:
            values = []
            for class_name in self.class_names:
                idx = self._csv_class_indices.get(class_name)
                if idx is None or idx >= len(row):
                    values.append(0)
                    continue
                try:
                    values.append(int(float(row[idx])))
                except ValueError:
                    values.append(0)

        return torch.tensor(values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.image_files)

    def build_binary_sample_weights(self) -> torch.Tensor | None:
        """Crea pesos para balancear clases en entrenamiento binario de casco."""
        if len(self.class_names) != 1:
            return None
        if self.label_source != "csv":
            return None

        labels = []
        class_name = self.class_names[0]
        class_idx = self._csv_class_indices.get(class_name)
        if class_idx is None:
            return None

        for img_path in self.image_files:
            row = self._csv_rows.get(img_path.name)
            if row is None or class_idx >= len(row):
                labels.append(0)
                continue
            try:
                labels.append(int(float(row[class_idx])))
            except ValueError:
                labels.append(0)

        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        if pos_count == 0 or neg_count == 0:
            return None

        total = len(labels)
        pos_weight = total / (2.0 * pos_count)
        neg_weight = total / (2.0 * neg_count)
        weights = [pos_weight if y == 1 else neg_weight for y in labels]
        return torch.tensor(weights, dtype=torch.double)
    
    def __getitem__(self, idx):

        from PIL import Image

        img_path = self.image_files[idx]

        # Cargar imagen
        image = Image.open(img_path).convert('RGB')

        if self.label_source == "csv":
            labels = self._labels_from_csv(img_path.name)
        else:
            label_path = self.data_dir / 'labels' / (img_path.stem + '.json')

            # Cargar etiquetas
            with open(label_path, 'r') as f:
                label_dict = json.load(f)

            if not isinstance(label_dict, dict):
                raise ValueError(f"Formato de label invalido en {label_path}")
            # Ignora labels fuera del objetivo (sin contar metadatos conocidos).
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
        max_finetune_epochs_without_improvement=1,
        freeze_backbone_epochs=0,
        max_train_batches_per_epoch=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.strict_no_cpu_fallback = strict_no_cpu_fallback
        self.patience = patience
        self.max_finetune_epochs_without_improvement = max(0, int(max_finetune_epochs_without_improvement))
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

            # Reporte único del dispositivo efectivo; la validación fuerte ya ocurre al inicio.
            if batch_idx == 0:
                model_device = next(self.model.parameters()).device
                print("[DEVICE] Runtime:")
                print(f"  - Modelo en: {model_device}")
                print(f"  - Inputs en: {inputs.device}")
                print(f"  - Labels en: {labels.device}")
            
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
        best_frozen_val_loss = float('inf')
        best_finetune_val_loss = float('inf')
        finetune_no_improve_counter = 0
        
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

            if epoch < self.freeze_backbone_epochs:
                best_frozen_val_loss = min(best_frozen_val_loss, val_loss)
            else:
                best_finetune_val_loss = min(best_finetune_val_loss, val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if epoch >= self.freeze_backbone_epochs:
                    finetune_no_improve_counter = 0
                
                if save_path:
                    self.model.save_model(save_path)
                    print(f"  [CHECKPOINT] Mejor modelo guardado")
            else:
                patience_counter += 1
                in_frozen_phase = epoch < self.freeze_backbone_epochs

                # Si la fase congelada se estanca, adelanta el fine-tuning
                # para comparar rapido si mejora al descongelar.
                if patience_counter >= self.patience and in_frozen_phase:
                    self.freeze_backbone_epochs = epoch + 1
                    patience_counter = 0
                    print(
                        "  [INFO] Sin mejora en fase congelada; "
                        "se adelanta fine-tuning para la siguiente época"
                    )
                    continue

                if not in_frozen_phase:
                    finetune_no_improve_counter += 1
                    if (
                        self.max_finetune_epochs_without_improvement > 0
                        and finetune_no_improve_counter >= self.max_finetune_epochs_without_improvement
                    ):
                        print(
                            "\n[EARLY STOPPING] Fine-tuning sin mejora; "
                            f"se detiene tras {finetune_no_improve_counter} época(s)"
                        )
                        break

                if patience_counter >= self.patience:
                    print(f"\n[EARLY STOPPING] Sin mejora en {self.patience} épocas")
                    break
        
        if best_finetune_val_loss < float('inf'):
            if best_finetune_val_loss < best_frozen_val_loss:
                print(
                    f"[OK] Fine-tuning mejoro Val Loss: "
                    f"{best_frozen_val_loss:.4f} -> {best_finetune_val_loss:.4f}"
                )
            else:
                print(
                    f"[INFO] Fine-tuning no mejoro Val Loss: "
                    f"mejor congelado={best_frozen_val_loss:.4f}, "
                    f"mejor fine-tuning={best_finetune_val_loss:.4f}"
                )

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
    model = ViTEPPClassifier(
        class_names=EPP_CLASS_NAMES,
        num_labels=ViT_TRAINING_CONFIG.get('num_labels'),
    )

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
        print(f"      {EPP_CSV_FILENAME}  (filename, Helmet, No Helmet)")
        print(f"      imagen_0000.jpg")
        print(f"      ...")
        print(f"    valid/   (o val/)")
        print(f"      {EPP_CSV_FILENAME}")
        print(f"      imagen_0500.jpg")
        print(f"      ...")
        print(f"    labels/  (legacy JSON)")
        print(f"      imagen_0000.json  ({{\"casco\": 1}})")
        print(f"      ...")
        return
    
    batch_size = ViT_TRAINING_CONFIG.get('batch_size', 32)
    if 'privateuseone' in str(device):
        # Tope conservador para estabilidad en DirectML (evita TDR/unknown error).
        directml_cap = ViT_TRAINING_CONFIG.get('directml_batch_cap', 4)
        if batch_size > directml_cap:
            print(f"[WARN] batch_size {batch_size} -> {directml_cap} para estabilidad DirectML")
            batch_size = directml_cap

    train_sampler = None
    if ViT_TRAINING_CONFIG.get('use_balanced_sampler', True):
        sample_weights = train_dataset.build_binary_sample_weights()
        if sample_weights is not None:
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            print("[FAST] Sampler balanceado activo (casco/sin casco)")

    if train_sampler is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    strict_no_cpu_fallback = ViT_TRAINING_CONFIG.get('strict_no_cpu_fallback', True)
    num_epochs = int(ViT_TRAINING_CONFIG.get('epochs', 20))
    fine_tune_last_epochs = max(0, int(ViT_TRAINING_CONFIG.get('fine_tune_last_epochs', 2)))
    freeze_backbone_epochs = max(0, num_epochs - fine_tune_last_epochs)
    max_train_batches_per_epoch = ViT_TRAINING_CONFIG.get('max_train_batches_per_epoch', 500)
    patience = ViT_TRAINING_CONFIG.get('patience', 5)
    max_finetune_epochs_without_improvement = ViT_TRAINING_CONFIG.get('max_finetune_epochs_without_improvement', 1)

    print(f"[FAST] Backbone congelado durante {freeze_backbone_epochs} épocas; fine-tuning en las últimas {fine_tune_last_epochs}")

    trainer = ViTEPPTrainer(
        model,
        device=device,
        lr=ViT_TRAINING_CONFIG.get('learning_rate', 1e-4),
        num_epochs=num_epochs,
        strict_no_cpu_fallback=strict_no_cpu_fallback,
        patience=patience,
        max_finetune_epochs_without_improvement=max_finetune_epochs_without_improvement,
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

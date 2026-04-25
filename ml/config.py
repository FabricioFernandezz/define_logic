DEVICE = 'directml'       # 'directml', 'cpu' 
VIDEO_SOURCE = 0          # 0 = webcam, o ruta a video

YOLO_MODEL_PATH = "yolov8n.pt"  # yolov8n/s/m/l para persona
YOLO_CONF_THRESHOLD = 0.5       # Umbral confianza YOLO
YOLO_IOU_THRESHOLD = 0.45       # IoU para NMS
YOLO_INFERENCE_DEVICE = "cpu"  # Etapa inicial estable: 'cpu' | luego 'directml'

ViT_MODEL_PATH = "models/models/vit_epp_best.pt"
ViT_PRETRAINED = "google/vit-base-patch16-224"
ViT_THRESHOLD = 0.7                          # Umbral clasificación (0-1), más estricto contra falsos positivos
ViT_MODEL_NAME = "google/vit-base-patch16-224"
ViT_INFERENCE_DEVICE = "cpu"                # Etapa inicial estable: 'cpu' | luego 'directml'
HELMET_HEAD_REGION_RATIO = 0.45             # Porción superior del bbox de persona usada para casco

# Etiquetas EPP activas (optimizadas a casco).
EPP_CLASS_NAMES = ["casco"]
EPP_HELMET_LABEL = "casco"
EPP_CSV_LABEL_MAP = {
    "helmet": "casco",
    "no helmet": "sin_casco",
    "no_helmet": "sin_casco",
}
EPP_CSV_FILENAME = "_classes.csv"

# Imagen estatica por defecto para inferencia inicial.
# Si no existe, run_static_image.py pedira una ruta por argumento.
STATIC_IMAGE_SOURCE = "assets/samples/imagen_prueba.png"

PIPELINE_CONFIG = {
    'yolo_model': YOLO_MODEL_PATH,
    'vit_model_path': ViT_MODEL_PATH,
    'device': DEVICE,
    'yolo_conf': YOLO_CONF_THRESHOLD,
    'vit_threshold': ViT_THRESHOLD,
    'crop_padding': 10,  # Píxeles de padding alrededor de personas
    'helmet_head_region_ratio': HELMET_HEAD_REGION_RATIO,
}

YOLO_DATASET_ROOT = "data/datasets/epp_unified"
EPP_DATASET_PATHS = ["data/datasets/epp_unified"]


ViT_DATASET_ROOT = r"C:\Users\Fabricio\OneDrive\Escritorio\Tesis\define_logic\data\datasets\cascos_dataset\datasets_download\zzz\Safety Helmet.v4-data160.multiclass"

YOLO_TRAINING_CONFIG = {
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'patience': 50,
    'device': 'directml',
}
EPP_DATA_YAML = "data/epp_data.yaml"

ViT_TRAINING_CONFIG = {
    'epochs': 20,
    'fine_tune_last_epochs': 2,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'dropout': 0.2,
    'patience': 5,           # Early stopping
    'max_finetune_epochs_without_improvement': 1,
    'use_balanced_sampler': True,
    'num_labels': len(EPP_CLASS_NAMES),
    'device': DEVICE,
    'strict_no_cpu_fallback': True,
    'directml_batch_cap': 6,
    'max_train_batches_per_epoch': 500,
}
RUNS_DIR = "runs"
INFERENCE_OUTPUT_DIR = f"{RUNS_DIR}/inference"
TRAINING_OUTPUT_DIR = f"{RUNS_DIR}/training"

# ROI poligonal (x, y) para la zona donde el casco es obligatorio.
# Ajusta estos puntos a tu escena real.
REQUIRED_ZONE_POLYGON = [
    (100, 80),
    (1180, 80),
    (1180, 680),
    (100, 680),
]

# Evidencias de personas en zona requerida sin casco.
AUDIT_OUTPUT_DIR = f"{RUNS_DIR}/audit"

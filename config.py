DEVICE = 'directml'       # 'directml', 'cpu' 
VIDEO_SOURCE = 0          # 0 = webcam, o ruta a video

YOLO_MODEL_PATH = "yolov8n.pt"  # yolov8n/s/m/l para persona
YOLO_CONF_THRESHOLD = 0.5       # Umbral confianza YOLO
YOLO_IOU_THRESHOLD = 0.45       # IoU para NMS

ViT_MODEL_PATH = "models/vit_epp_best.pt"    # Modelo ViT entrenado
ViT_SIMPLE_MODEL_PATH = "models/vit_epp_simple_best.pt"  # Fase 1 (backbone congelado)
ViT_PRETRAINED = "google/vit-base-patch16-224"  # Modelo base
ViT_THRESHOLD = 0.5                          # Umbral clasificación (0-1)
ViT_MODEL_NAME = "google/vit-base-patch16-224"

PIPELINE_CONFIG = {
    'yolo_model': YOLO_MODEL_PATH,
    'vit_model_path': ViT_MODEL_PATH,
    'device': DEVICE,
    'yolo_conf': YOLO_CONF_THRESHOLD,
    'vit_threshold': ViT_THRESHOLD,
    'crop_padding': 10,  # Píxeles de padding alrededor de personas
}

YOLO_DATASET_ROOT = "data/datasets/epp_unified"
EPP_DATASET_PATHS = ["data/datasets/epp_unified"]


ViT_DATASET_ROOT = r"C:\Users\Fabricio\OneDrive\Escritorio\Tesis\define_logic\data\datasets\cascos_dataset\datasets_download\Safety Helmet.v4-data160.multiclass"

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
    'simple_epochs': 5,
    'simple_freeze_all_backbone': True,
    'phase2_init_from_simple': True,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'dropout': 0.2,
    'patience': 5,           # Early stopping
    'num_labels': 3,         # casco, guantes, chaleco
    'device': DEVICE,
    'strict_no_cpu_fallback': True,
    'directml_batch_cap': 6,
    'freeze_backbone_epochs': 3,
    'max_train_batches_per_epoch': 800,
}
RUNS_DIR = "runs"
INFERENCE_OUTPUT_DIR = f"{RUNS_DIR}/inference"
TRAINING_OUTPUT_DIR = f"{RUNS_DIR}/training"

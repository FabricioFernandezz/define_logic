# Configuración de detección de personas
MODEL_PATH = "yolov8n.pt"
VIDEO_SOURCE = 0   # webcam
CONF_THRESHOLD = 0.5

# Configuración de EPP Classifier (Cascos, Guantes, Chalecos)
EPP_MODEL_PATH = "models/epp_best.pt"  # Ruta al modelo entrenado

# Rutas de los datasets para entrenamiento
# IMPORTANTE: Reemplaza estas rutas con las rutas reales de tus datasets
EPP_DATASET_PATHS = [
    "data/datasets/cascos_dataset",    # Dataset de cascos
    "data/datasets/guantes_dataset",   # Dataset de guantes
    "data/datasets/chalecos_dataset"   # Dataset de chalecos
]

# Si tienes un dataset unificado con todas las clases, usa solo una ruta:
# EPP_DATASET_PATHS = ["data/datasets/epp_unified"]

# Configuración de entrenamiento EPP
EPP_TRAINING_CONFIG = {
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'patience': 50,
    'conf_threshold': 0.5,
    'device': '',  # '' = auto-detect GPU/CPU, '0' = GPU 0, 'cpu' = CPU
}

# Archivo YAML de configuración del dataset
EPP_DATA_YAML = "data/epp_data.yaml"

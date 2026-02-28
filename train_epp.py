"""
Script de ejemplo para entrenar el clasificador EPP
Detecta: cascos, guantes y chalecos de seguridad
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from models.epp_classifier import EPPClassifier, create_data_yaml
from config import EPP_DATASET_PATHS, EPP_TRAINING_CONFIG, EPP_DATA_YAML


def main():
    """
    Flujo completo de entrenamiento del modelo EPP
    
    Pasos:
    1. Crear archivo YAML de configuración del dataset
    2. Inicializar el clasificador
    3. Entrenar el modelo
    4. Validar el modelo
    5. Guardar métricas
    """
    
    print("=" * 60)
    print("ENTRENAMIENTO DEL CLASIFICADOR EPP")
    print("Detección de: Cascos, Guantes, Chalecos")
    print("=" * 60)
    
    # Paso 1: Crear configuración del dataset
    print("\n[1/4] Creando archivo de configuración del dataset...")
    try:
        yaml_path = create_data_yaml(EPP_DATASET_PATHS, EPP_DATA_YAML)
        print(f"✓ Configuración creada: {yaml_path}")
        print("\n⚠️  IMPORTANTE: Edita el archivo YAML con las rutas correctas de tu dataset")
        print(f"   Archivo: {yaml_path}")
        print("   Asegúrate de que la estructura sea:")
        print("   dataset/")
        print("   ├── images/")
        print("   │   ├── train/")
        print("   │   ├── val/")
        print("   │   └── test/")
        print("   └── labels/")
        print("       ├── train/")
        print("       ├── val/")
        print("       └── test/")
    except Exception as e:
        print(f"✗ Error al crear configuración: {e}")
        return
    
    # Paso 2: Inicializar clasificador
    print(f"\n[2/4] Inicializando clasificador EPP...")
    try:
        classifier = EPPClassifier()
        info = classifier.get_model_info()
        print(f"✓ Clasificador inicializado")
        print(f"   Tipo: {info['type']}")
        print(f"   Clases: {list(info['names'].values())}")
        print(f"   Device: {info['device']}")
    except Exception as e:
        print(f"✗ Error al inicializar clasificador: {e}")
        return
    
    # Paso 3: Entrenar modelo
    print(f"\n[3/4] Entrenando modelo...")
    print(f"   Épocas: {EPP_TRAINING_CONFIG['epochs']}")
    print(f"   Batch size: {EPP_TRAINING_CONFIG['batch']}")
    print(f"   Imagen size: {EPP_TRAINING_CONFIG['imgsz']}")
    print(f"   Device: {EPP_TRAINING_CONFIG['device'] or 'auto'}")
    print("\n⚠️  El entrenamiento puede tardar varias horas dependiendo del hardware")
    print("   Presiona Ctrl+C para cancelar\n")
    
    try:
        results = classifier.train(
            data_yaml_path=yaml_path,
            epochs=EPP_TRAINING_CONFIG['epochs'],
            imgsz=EPP_TRAINING_CONFIG['imgsz'],
            batch=EPP_TRAINING_CONFIG['batch'],
            patience=EPP_TRAINING_CONFIG['patience'],
            device=EPP_TRAINING_CONFIG['device']
        )
        
        print(f"\n✓ Entrenamiento completado exitosamente")
        print(f"   Modelo guardado en: {results.save_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento cancelado por el usuario")
        return
    except Exception as e:
        print(f"\n✗ Error durante entrenamiento: {e}")
        return
    
    # Paso 4: Validar modelo
    print(f"\n[4/4] Validando modelo...")
    try:
        metrics = classifier.validate(yaml_path)
        
        print("\n" + "=" * 60)
        print("RESULTADOS FINALES")
        print("=" * 60)
        print(f"mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"Precision    : {metrics.box.mp:.4f}")
        print(f"Recall       : {metrics.box.mr:.4f}")
        print("=" * 60)
        
        # Recomendaciones basadas en métricas
        print("\n📊 Análisis de resultados:")
        if metrics.box.map50 > 0.9:
            print("   ✓ Excelente rendimiento (mAP50 > 0.9)")
        elif metrics.box.map50 > 0.7:
            print("   ✓ Buen rendimiento (mAP50 > 0.7)")
        elif metrics.box.map50 > 0.5:
            print("   ⚠️  Rendimiento moderado (mAP50 > 0.5)")
            print("   💡 Considera aumentar las épocas o mejorar el dataset")
        else:
            print("   ✗ Rendimiento bajo (mAP50 < 0.5)")
            print("   💡 Revisa la calidad del dataset y las anotaciones")
        
    except Exception as e:
        print(f"⚠️  Error durante validación: {e}")
        print("   El modelo fue entrenado pero no se pudo validar")
    
    print("\n✓ Proceso completado")
    print(f"\n💾 Para usar el modelo entrenado, actualiza EPP_MODEL_PATH en config.py")
    print(f"   con la ruta: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()

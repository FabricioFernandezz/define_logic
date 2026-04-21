# ML Workspace

Esta carpeta centraliza los entrypoints de entrenamiento y utilidades de datos.

## Entrenamiento

```bash
python -m ml.training.train_vit_epp
python -m ml.training.train_vit_epp_simple
```

## Herramientas

```bash
python -m ml.tools.yolo_to_vit_converter
python -m ml.tools.reorganize_labels
python -m ml.tools.roboflow_csv_to_json_converter_v2
```

Estos entrypoints delegan a los scripts originales para mantener compatibilidad con el código existente.

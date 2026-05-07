from ultralytics import YOLO

# Carga el modelo ONNX exportado
model_onnx = YOLO(r"C:\Users\Fabricio\OneDrive\Escritorio\Tesis\define_logic\ml\runs\yolo26_epp\best.onnx")  # Ruta a tu archivo .onnx

# Carga el modelo ONNX exportado

# Realiza la inferencia utilizando el modelo ONNX
# Puedes usar la misma imagen de prueba o cualquier otra imagen
# Para detectar solo ciertas clases, especifica sus IDs en la lista 'classes'.
# Por ejemplo: classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] para todas las clases.
# O, elimina el argumento 'classes' para detectar todas por defecto.
# Clases disponibles:
#   0: helmet
#   1: gloves
#   2: vest
#   3: boots
#   4: goggles
#   5: none
#   6: Person
#   7: no_helmet
#   8: no_goggle
#   9: no_gloves
#  10: no_boots

prediction_results_onnx = model_onnx.predict(r"C:\Users\Fabricio\Downloads\test\EDITADA-scaled-1.jpg", save=True, classes=[0, 2, 4, 6])

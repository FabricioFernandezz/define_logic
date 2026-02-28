from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        detections = []

        for r in results:  # recorremos la lista de imagenes
            for box in r.boxes:  # recorremos cada uno de estos boxes
                cls = int(box.cls[0])  # obtenemos el numero de clase
                conf = float(box.conf[0])  # obtenemos la confianza

                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # obtenemos las coordenadas del box
                    detections.append({  # agregamos la deteccion a la lista de detecciones
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf
                    })

        return detections



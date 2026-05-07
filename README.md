## **Proyecto Final** - DefineLogic
* Dibuja las reglas. 
* Define el riesgo. 
* Protege a tu Equipo.

*Autor: Fabricio Fernández*
---

 Después de 10 épocas revisá las métricas en ml/runs/epp_general*/results.csv — si box_loss baja y mAP sube, el modelo está aprendiendo. Si los resultados son prometedores, podés continuar desde el último   
  checkpoint con --resume.

  # Ultra rápido: modelo nano, 10% datos, 2 épocas (~5-8 min total)                                                                                                                                             
  python ml/training/train_yolo_epp.py --model yolov8n.pt --epochs 2 --batch 4 --fraction 0.1
                                                                                                                                                                                                                
  # Rápido: modelo small, 20% datos, 3 épocas               
  python ml/training/train_yolo_epp.py --model yolov8s.pt --epochs 3 --batch 4 --fraction 0.2

  # Tu setup actual (1h por época)
  python ml/training/train_yolo_epp.py --model yolov8m.pt --epochs 10 --batch 4


venv\Scripts\python ml\training\train_yolo_epp.py --epochs 10 --batch 4 --device 0 --workers 0 --fraction 1.0 --no-val --name epp_10ep

  venv\Scripts\python -m ultralytics yolo val model=ml\runs\epp_10ep\weights\best.pt data=data\datasets\cascos_dataset\datasets_download\dataset_general_yolo\data.yaml device=cpu

  después de entrenar, revisá carpeta ml\runs\epp_10ep\weights\

# ENTRENAMIENTO CORTOOOO

  venv\Scripts\python ml\training\train_yolo_epp.py --model yolov8n.pt --epochs 2 --batch 4 --fraction 0.1 --device 0 --workers 0 --no-val

  DEBERIA PROBAR CON 6 EN EL BATCH?

  venv\Scripts\python ml\training\train_yolo_epp2.py --model yolov8n.pt --epochs 2 --batch 4 --fraction 0.1 --device 0 --workers


yolov8n.pt (nano): pruebas rápidas, poco VRAM, baja precisión. Útil para debug y ajuste de pipeline.
yolov8s.pt (small): buen compromiso; recomendado para la mayoría de experiments iniciales.
yolov8m.pt (medium): mejor precisión pero mayor consumo (GPU+tiempo). Úsalo para runs finales si los resultados con s son prometedores.

"C:\Users\Fabricio\Downloads\test\trabajador-con-casco-seguridad.jpg"




# ~10min/época → 50 épocas ≈ 8h
python ml/training/train_yolo_epp2.py --model yolov8m.pt --epochs 50 --batch 4 --no-val --fraction 0.1 --imgsz 320 --name epp_altec_50ep_rapido

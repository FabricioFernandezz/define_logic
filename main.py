"""Entry point principal: pipeline YOLO (personas) -> ViT (EPP)."""

import cv2
from pathlib import Path
import time

from config import PIPELINE_CONFIG, VIDEO_SOURCE
from models.vision_pipeline import EPPVisionPipeline


def main() -> None:
    print("=" * 70)
    print("DEFINE LOGIC - PIPELINE YOLO -> ViT")
    print("=" * 70)

    pipeline = EPPVisionPipeline(
        yolo_model=PIPELINE_CONFIG["yolo_model"],
        vit_model_path=PIPELINE_CONFIG.get("vit_model_path"),
        device=PIPELINE_CONFIG.get("device", "directml"),
        yolo_conf=PIPELINE_CONFIG.get("yolo_conf", 0.5),
        vit_threshold=PIPELINE_CONFIG.get("vit_threshold", 0.5),
        crop_padding=PIPELINE_CONFIG.get("crop_padding", 10),
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {VIDEO_SOURCE}")

    frame_count = 0
    fps_window = []

    print("[INFO] Presiona 'q' para salir, 's' para guardar frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de video o error de captura.")
            break

        frame_count += 1
        t0 = time.time()

        results = pipeline.process_frame(frame, return_crops=False, input_is_bgr=True)

        for person in results["persons"]:
            x1, y1, x2, y2 = person["bbox_pixels"]
            epp = person["epp"]
            has_all_epp = all(epp[item]["present"] for item in ["casco", "guantes", "chaleco"])
            color = (0, 255, 0) if has_all_epp else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"P{person['person_id']} det:{person['detection_conf']:.2f}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        dt = time.time() - t0
        fps_window.append(1.0 / dt if dt > 0 else 0)
        if len(fps_window) > 30:
            fps_window.pop(0)
        fps = sum(fps_window) / len(fps_window)

        cv2.putText(
            frame,
            f"FPS:{fps:.1f} Personas:{results['num_persons']}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("DefineLogic YOLO->ViT", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out = Path("runs/inference") / f"frame_{frame_count:06d}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame)
            print(f"[OK] Frame guardado: {out}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

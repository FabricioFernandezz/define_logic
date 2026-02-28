from models.person_detector import PersonDetector
from video.capture import Capture
from utils.drawing import Draw
from utils.fps import FPS
import config
import cv2 as cv

def main():
    person_detector = PersonDetector(config.MODEL_PATH)
    video_capture = Capture(config.VIDEO_SOURCE)
    fps = FPS().start()

    while True:
        try:
            frame = video_capture.read()
            detections = person_detector.detect(frame)

            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                confidence = detection["confidence"]
                Draw.rectangle(frame, x1, y1, x2, y2)
                Draw.text(frame, f"{confidence:.2f}", x1, y1 - 10)

            fps.update()
            cv.imshow("Person Detection", frame)

            if cv.waitKey(1) & 0xFF == ord('d'):
                break
        except RuntimeError as e:
            print(e)
            break

    # Detener el medidor de FPS y liberar los recursos
    fps.stop()
    print(f"Elapsed time: {fps.elapsed():.2f} seconds")
    print(f"Approximate FPS: {fps.fps():.2f}")
    video_capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
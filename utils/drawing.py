import cv2 as cv

class Draw:
    @staticmethod
    def rectangle(frame, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
        cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def text(frame, text, x, y, font=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), thickness=1):
        cv.putText(frame, text, (x, y), font, font_scale, color, thickness)
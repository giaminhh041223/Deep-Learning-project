import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bbox_landmarks(img, bbox, kps, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    for (x, y) in kps:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)


def draw_label(img, bbox, text, color=(0, 200, 255)):
    x1, y1, x2, y2 = map(int, bbox)
    label = text
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
    th += 6
    box_end = (x1 + tw + 8, y1 - 10 + th)
    cv2.rectangle(img, (x1, y1 - 10), box_end, color, -1)
    cv2.putText(img, label, (x1 + 4, y1 + th - 12), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def draw_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
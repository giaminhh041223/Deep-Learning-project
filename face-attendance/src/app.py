import argparse
import cv2
import numpy as np
from .config import CFG
from .utils import FPS
from .draw import draw_bbox_landmarks, draw_label, draw_fps
from .detection import Detector
from .alignment import Aligner
from .embedding import Embedder
from .matching import FaceBank


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=str, default='0', help='0 webcam, path, or URL')
    ap.add_argument('--threshold', type=float, default=CFG.threshold)
    ap.add_argument('--min_face', type=int, default=CFG.min_face)
    ap.add_argument('--draw-fps', type=int, default=int(CFG.draw_fps))
    return ap.parse_args()


def open_source(src: str):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
        
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")
        
    return cap


def main():
    args = build_args()

    cap = open_source(args.source)

    detector = Detector()
    aligner = Aligner(112)
    embedder = Embedder()
    facebank = FaceBank()

    fps = FPS()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        img = frame.copy()
        dets = detector.detect(img, min_face=args.min_face)

        for d in dets:
            bbox = d['bbox']
            kps = d['kps']
            
            crop = aligner.align(img, kps)
            emb = embedder.embed(crop)
            label, score, _ = facebank.match(emb, threshold=args.threshold)

            draw_bbox_landmarks(img, bbox, kps)
            draw_label(img, bbox, f"{label} ({score:.2f})")

        if args.draw_fps:
            draw_fps(img, fps.tick())

        cv2.imshow('Face Attendance — Realtime', img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
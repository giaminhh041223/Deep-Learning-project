import cv2
import argparse
from insightface.app import FaceAnalysis
from .aligner import FaceAligner
from .utils import show_side_by_side, draw_landmarks

def main(img_path):
    # --- Load image ---
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        return

    # --- Face detection ---
    app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection'])
    app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.6)
    faces = app.get(img)
    if not faces:
        print("[WARN] No face detected.")
        return

    face = faces[0]
    landmarks = face.kps

    # --- Draw landmarks for visualization ---
    img_marked = draw_landmarks(img, landmarks)

    # --- Face alignment ---
    aligner = FaceAligner()
    aligned = aligner.align(img, landmarks)

    if aligned is None:
        print("[WARN] Alignment failed.")
        return

    # --- Show result ---
    show_side_by_side(img_marked, aligned)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to input image")
    args = parser.parse_args()
    main(args.img)

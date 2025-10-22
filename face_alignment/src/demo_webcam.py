import cv2
import time
from insightface.app import FaceAnalysis
from .aligner import FaceAligner

def main():
    app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection'])
    app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.6)

    aligner = FaceAligner()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            lmk = face.kps
            aligned = aligner.align(frame, lmk)
            if aligned is not None:
                cv2.imshow("Aligned", aligned)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

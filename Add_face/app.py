# app.py
import os
import argparse
import pickle
import cv2
import numpy as np

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model không tìm thấy: {model_path}. Hãy chạy train_model.py trước.")
    with open(model_path, "rb") as f:
        knn = pickle.load(f)
    return knn

def main(camera_idx=0, threshold=3000.0):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    model_path = os.path.join(DATA_DIR, "face_recognition_model.pkl")

    print("[INFO] Loading model...")
    knn = load_model(model_path)
    print("[INFO] Model loaded.")

    # Cascade (dùng file trong project nếu có, else cv2 default)
    cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("[ERROR] Không mở được camera.")
        return

    print("Ấn 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # vẽ khung
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            # tiền xử lý giống lúc train
            face = gray[y:y+h, x:x+w]
            try:
                face = cv2.resize(face, (50, 50))
            except Exception:
                continue
            feat = face.flatten().reshape(1, -1)  # (1, 2500)

            # Lấy khoảng cách tới k neighbors
            dists, idxs = knn.kneighbors(feat, n_neighbors=min(3, knn.n_neighbors), return_distance=True)
            mean_dist = np.mean(dists)

            if mean_dist < threshold:
                name = knn.predict(feat)[0]
            else:
                name = "Unknown"

            label = f"{name} ({mean_dist:.0f})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--threshold", type=float, default=3000.0, help="distance threshold for unknown")
    args = parser.parse_args()
    main(camera_idx=args.camera, threshold=args.threshold)

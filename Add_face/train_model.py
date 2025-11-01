import cv2
import os
import numpy as np
import pickle
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

DATA_DIR = "data"
DATASET_DIR = "dataset"

def preprocess_image(img_path):
    """Đọc, resize, normalize ảnh"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (50, 50))
    img = img.astype("float32") / 255.0  # normalize
    return img.flatten()

def remove_duplicates(faces, names):
    """Loại bỏ ảnh trùng lặp"""
    unique_faces, unique_names = [], []
    seen = set()
    for face, name in zip(faces, names):
        h = hash(face.tobytes())
        if h not in seen:
            seen.add(h)
            unique_faces.append(face)
            unique_names.append(name)
    return np.array(unique_faces), np.array(unique_names)

def load_from_dataset():
    """Duyệt dataset folder"""
    faces, names = [], []
    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue
        print(f"[INFO] Processing {person_name} ...")
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            feat = preprocess_image(img_path)
            if feat is not None:
                faces.append(feat)
                names.append(person_name)

    faces, names = remove_duplicates(faces, names)
    faces, names = shuffle(faces, names, random_state=42)

    print(f"[INFO] Final dataset: {len(names)} samples, {len(set(names))} persons")
    return faces, names

def train_knn(faces, names):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, names)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "face_recognition_model.pkl"), "wb") as f:
        pickle.dump(knn, f)
    print("[INFO] Model trained & saved at data/face_recognition_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-dataset", action="store_true",
                        help="Train model từ dataset/ thay vì pickle")
    args = parser.parse_args()

    if args.from_dataset:
        faces, names = load_from_dataset()
    else:
        with open(os.path.join(DATA_DIR, "faces_data.pkl"), "rb") as f:
            faces = pickle.load(f)
        with open(os.path.join(DATA_DIR, "names.pkl"), "rb") as f:
            names = pickle.load(f)

    train_knn(faces, names)

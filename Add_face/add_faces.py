import cv2
import numpy as np
import os
import pickle

# Tạo thư mục data nếu chưa có
if not os.path.exists("data"):
    os.makedirs("data")

# Load dữ liệu cũ nếu có
if os.path.exists("data/faces_data.pkl") and os.path.exists("data/names.pkl"):
    with open("data/faces_data.pkl", "rb") as f:
        faces_data = pickle.load(f)
        if isinstance(faces_data, np.ndarray):
            faces_data = faces_data.tolist()

    with open("data/names.pkl", "rb") as f:
        names = pickle.load(f)
        if isinstance(names, np.ndarray):
            names = names.tolist()
else:
    faces_data = []
    names = []

# Nhập tên người dùng
person_name = input("Nhập tên người: ").strip()

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
print("Ấn Q để thoát...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (50, 50))  # resize cố định

        # Mỗi 5 frame lưu 1 ảnh để tránh trùng
        if count % 5 == 0:
            faces_data.append(face.flatten())   # (2500,)
            names.append(person_name)

        count += 1

    cv2.imshow("Thêm khuôn mặt", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Convert sang numpy array
faces_data = np.array(faces_data)  # (N, 2500)

# Lưu lại
with open("data/faces_data.pkl", "wb") as f:
    pickle.dump(faces_data, f)

with open("data/names.pkl", "wb") as f:
    pickle.dump(names, f)

print("✅ Đã lưu dữ liệu:", faces_data.shape, "số label:", len(names))

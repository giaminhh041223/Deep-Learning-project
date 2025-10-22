import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_side_by_side(img1, img2, title1="Original", title2="Aligned"):
    """Hiển thị 2 ảnh cạnh nhau."""
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_rgb)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2_rgb)
    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def draw_landmarks(img, landmarks, color=(0, 255, 0)):
    """Vẽ các điểm landmarks lên ảnh."""
    out = img.copy()
    for (x, y) in landmarks:
        cv2.circle(out, (int(x), int(y)), 2, color, -1)
    return out

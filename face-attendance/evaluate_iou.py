import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from src.detection import Detector
from src.config import CFG

# --- CẤU HÌNH ĐƯỜNG DẪN (Dựa trên ảnh bạn gửi) ---
# Chúng ta dùng tập 'val' để đánh giá
IMG_DIR = Path("archive/images/val") 
LABEL_DIR = Path("archive/labels/val")

def yolo_to_bbox(yolo_line, img_w, img_h):
    """
    Chuyển đổi YOLO (x_center, y_center, w, h) -> (x1, y1, x2, y2)
    """
    parts = yolo_line.strip().split()
    # parts[0] là class_id, ta bỏ qua
    if len(parts) < 5: return None
    
    x_c, y_c, w, h = map(float, parts[1:5])
    
    x1 = int((x_c - w/2) * img_w)
    y1 = int((y_c - h/2) * img_h)
    x2 = int((x_c + w/2) * img_w)
    y2 = int((y_c + h/2) * img_h)
    return [x1, y1, x2, y2]

def compute_iou(boxA, boxB):
    # 1. Tìm tọa độ giao nhau
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 2. Tính diện tích giao
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # 3. Tính diện tích từng box
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 4. Tính IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate():
    print("🚀 Bắt đầu đánh giá IoU trên tập Validation...")
    
    if not IMG_DIR.exists() or not LABEL_DIR.exists():
        print(f"❌ Không tìm thấy thư mục 'val'.")
        print(f"👉 Kiểm tra lại: {IMG_DIR}")
        return

    # Khởi tạo Detector (Dùng CPU để tránh lỗi crash trên Mac)
    detector = Detector()
    
    # Lấy danh sách ảnh trong folder val
    img_files = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.jpeg")) + list(IMG_DIR.glob("*.png"))
    print(f"📂 Tìm thấy {len(img_files)} ảnh trong tập Validation.")
    
    ious = []
    tp, fp, fn = 0, 0, 0
    
    print("⚙️ Đang xử lý...")
    for img_path in tqdm(img_files):
        # Tìm file label tương ứng (cùng tên, đuôi .txt) trong folder labels/val
        label_path = LABEL_DIR / (img_path.stem + ".txt")
        
        # Đọc ảnh
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img = img.shape[:2]

        # Đọc Ground Truth (YOLO format)
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    box = yolo_to_bbox(line, w_img, h_img)
                    if box: gt_boxes.append(box)
        
        # Nếu ảnh không có file label -> coi như không có mặt (Ground Truth rỗng)
        
        # Detect bằng model của chúng ta
        faces = detector.detect(img)
        pred_boxes = [f['bbox'].astype(int) for f in faces]

        # Tính toán IoU
        matched_gt = set()
        for p_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                iou = compute_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Ngưỡng IoU > 0.5 là chuẩn
            if best_iou >= 0.5:
                if best_gt_idx not in matched_gt:
                    tp += 1
                    ious.append(best_iou)
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1 # Trùng lặp
            else:
                fp += 1 # Detect sai vị trí
        
        fn += (len(gt_boxes) - len(matched_gt))

    # Tổng kết
    avg_iou = np.mean(ious) if ious else 0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    print("\n" + "="*40)
    print("📊 KẾT QUẢ ĐÁNH GIÁ (FACE DETECTION DATASET)")
    print("="*40)
    print(f"🔹 Average IoU: {avg_iou:.4f}")
    print("-" * 40)
    print(f"🔹 Precision: {precision:.2%}")
    print(f"🔹 Recall: {recall:.2%}")
    print(f"🔹 F1-Score: {f1:.2%}")
    print("="*40)

if __name__ == "__main__":
    evaluate()

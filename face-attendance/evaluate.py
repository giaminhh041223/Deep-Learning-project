import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.detection import Detector
from src.alignment import Aligner
from src.embedding import Embedder
from src.matching import FaceBank
from src.config import CFG

# Tạo thư mục lưu ảnh lỗi
ERROR_DIR = Path("evaluation_errors")
ERROR_DIR.mkdir(exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, labels):
    """Vẽ và lưu Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("📈 Đã lưu biểu đồ confusion_matrix.png")

def evaluate(test_dir="data/test_dataset"):
    print("🚀 Bắt đầu đánh giá chuyên sâu...")
    
    detector = Detector()
    aligner = Aligner()
    embedder = Embedder()
    facebank = FaceBank()

    if facebank.embeddings is None:
        print("❌ FaceBank rỗng. Hãy chạy register.py trước.")
        return

    img_paths = []
    true_labels = []
    root = Path(test_dir)
    
    # Duyệt thư mục test
    for folder in root.iterdir():
        if folder.is_dir():
            for img in folder.glob("*"):
                img_paths.append(str(img))
                true_labels.append(folder.name)

    pred_labels = []
    valid_true_labels = [] 
    
    detected_count = 0
    
    print(f"⚙️ Config: Threshold={CFG.threshold} | Min Face={CFG.min_face}")

    for img_path, true_label in tqdm(zip(img_paths, true_labels), total=len(img_paths)):
        img = cv2.imread(img_path)
        if img is None: continue
        
        debug_img = img.copy() # Dùng để vẽ lỗi

        # --- DETECTION ---
        faces = detector.detect(img)
        
        if len(faces) == 0:
            pred_labels.append("Unknown")
            valid_true_labels.append(true_label)
            
            # Lưu ảnh lỗi không tìm thấy mặt
            cv2.imwrite(str(ERROR_DIR / f"NoFace_{Path(img_path).name}"), img)
            continue
        
        detected_count += 1
        target_face = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]), reverse=True)[0]
        
        # --- RECOGNITION ---
        crop = aligner.align(img, target_face['kps'])
        emb = embedder.embed(crop)
        
        pred_label_raw, score, _ = facebank.match(emb, threshold=CFG.threshold)
        pred_name = pred_label_raw.split(" | ")[0] if pred_label_raw != "Unknown" else "Unknown"
        
        pred_labels.append(pred_name)
        valid_true_labels.append(true_label)

        # --- ERROR ANALYSIS ---
        # Nếu nhận diện sai hoặc Unknown (trong khi ground truth là người đã biết)
        if pred_name != true_label:
            # Vẽ box và tên dự đoán sai lên ảnh
            bbox = target_face['bbox'].astype(int)
            cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(debug_img, f"Pred: {pred_name} ({score:.2f})", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(debug_img, f"True: {true_label}", (bbox[0], bbox[1]-35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Lưu ảnh lỗi
            filename = f"Wrong_{true_label}_as_{pred_name}_{Path(img_path).name}"
            cv2.imwrite(str(ERROR_DIR / filename), debug_img)

    # --- REPORT ---
    print("\n" + "="*50)
    print("📊 BÁO CÁO KẾT QUẢ")
    
    # Accuracy
    acc = accuracy_score(valid_true_labels, pred_labels)
    print(f"✅ Accuracy: {acc:.2%}")
    
    # Classification Report
    unique_labels = sorted(list(set(valid_true_labels + pred_labels)))
    print("\n📋 Chi tiết:")
    print(classification_report(valid_true_labels, pred_labels, zero_division=0))
    
    # Vẽ Confusion Matrix
    try:
        plot_confusion_matrix(valid_true_labels, pred_labels, unique_labels)
    except Exception as e:
        print(f"⚠️ Không thể vẽ Confusion Matrix: {e}")

    print(f"\n📂 Đã lưu các ảnh nhận diện sai vào thư mục: {ERROR_DIR}")
    print("="*50)

if __name__ == "__main__":
    evaluate()

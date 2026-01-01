import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

from src.detection import Detector
from src.alignment import Aligner
from src.embedding import Embedder
from src.matching import FaceBank

def batch_register(data_dir="data/to_register"):
    print("🚀 Đang nạp khuôn mặt vào FaceBank...")
    
    detector = Detector()
    aligner = Aligner()
    embedder = Embedder()
    
    # Xóa FaceBank cũ
    facebank_path = Path("data/facebank")
    if facebank_path.exists(): shutil.rmtree(facebank_path)
    facebank = FaceBank()
    
    root = Path(data_dir)
    if not root.exists():
        print("❌ Chưa chạy setup data!")
        return

    person_dirs = [p for p in root.iterdir() if p.is_dir()]
    
    count_person = 0
    
    for person_dir in tqdm(person_dirs):
        name = person_dir.name
        img_files = list(person_dir.glob("*"))
        
        has_face = False
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is None: continue

            faces = detector.detect(img)
            if len(faces) == 0: continue
            
            # Lấy mặt lớn nhất
            target_face = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]), reverse=True)[0]
            
            crop = aligner.align(img, target_face['kps'])
            emb = embedder.embed(crop)
            
            facebank.add(emb, name=name)
            has_face = True
        
        if has_face:
            count_person += 1

    facebank.save()
    print(f"\n✅ Đã đăng ký thành công {count_person} người nổi tiếng vào hệ thống.")

if __name__ == "__main__":
    batch_register()

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# CẤU HÌNH
# Đổi tên folder dưới đây cho đúng với tên folder bạn giải nén
SOURCE_ROOT = Path("Celebrity Faces Dataset") 

DEST_REGISTER = Path("data/to_register")
DEST_EVALUATE = Path("data/test_dataset")

# Số lượng ảnh dùng để đăng ký (Học) cho mỗi người
# Vì mỗi người có ~100 ảnh, lấy 10 ảnh để học là rất tốt.
NUM_REGISTER = 10 

def setup_data():
    if not SOURCE_ROOT.exists():
        print(f"❌ Không tìm thấy thư mục: {SOURCE_ROOT}")
        print("👉 Hãy chắc chắn bạn đã giải nén và đặt đúng tên thư mục.")
        return

    # Dọn dẹp thư mục cũ
    if DEST_REGISTER.exists(): shutil.rmtree(DEST_REGISTER)
    if DEST_EVALUATE.exists(): shutil.rmtree(DEST_EVALUATE)
    
    DEST_REGISTER.mkdir(parents=True)
    DEST_EVALUATE.mkdir(parents=True)

    print(f"🚀 Bắt đầu chia dữ liệu từ: {SOURCE_ROOT}")
    
    # Duyệt qua từng người nổi tiếng
    people_dirs = [p for p in SOURCE_ROOT.iterdir() if p.is_dir()]
    
    for person_dir in tqdm(people_dirs):
        person_name = person_dir.name
        
        # Lấy danh sách ảnh
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
        
        # Xáo trộn ngẫu nhiên để không lấy phải các ảnh giống hệt nhau (nếu là cắt từ video)
        random.shuffle(images)
        
        # Kiểm tra nếu ít ảnh quá
        if len(images) < 2:
            continue # Bỏ qua người này

        # Tính toán điểm cắt
        # Nếu tổng ảnh < 15, thì chỉ lấy 1 ảnh đăng ký, còn lại test
        n_reg = NUM_REGISTER if len(images) > 15 else 1
        
        register_imgs = images[:n_reg]
        test_imgs = images[n_reg:]
        
        # --- COPY ẢNH ĐĂNG KÝ ---
        reg_dest = DEST_REGISTER / person_name
        reg_dest.mkdir(parents=True, exist_ok=True)
        for img in register_imgs:
            shutil.copy(str(img), str(reg_dest / img.name))
            
        # --- COPY ẢNH KIỂM THỬ ---
        test_dest = DEST_EVALUATE / person_name
        test_dest.mkdir(parents=True, exist_ok=True)
        for img in test_imgs:
            shutil.copy(str(img), str(test_dest / img.name))

    print("\n✅ Hoàn tất!")
    print(f"📂 Dữ liệu học: {DEST_REGISTER} (Mỗi người {NUM_REGISTER} ảnh)")
    print(f"📂 Dữ liệu thi: {DEST_EVALUATE}")

if __name__ == "__main__":
    setup_data()
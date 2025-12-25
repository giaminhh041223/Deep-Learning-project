import os
from pathlib import Path
from src.backend_manager import register_user, create_class, add_student_to_class

def init_system():
    print("[INIT] Đang khởi tạo hệ thống dữ liệu...")
    
    #Tạo thư mục
    base_dir = Path("data")
    (base_dir / "classes").mkdir(parents=True, exist_ok=True)
    (base_dir / "facebank").mkdir(parents=True, exist_ok=True) 
    
    #Tạo tài khoản Admin mặc định
    if register_user("admin", "00000"):
        print("--> Đã tạo tài khoản: admin")
    else:
        print("--> Tài khoản admin đã tồn tại.")
        
    #Tạo một lớp học mẫu
    class_id = create_class("IT3320E", "Introduction to Deep Learning", 15)
    print(f"--> Đã tạo lớp mẫu: IT3030 (ID: {class_id})")
    
    #Thêm vài sinh viên mẫu vào lớp đó
    students = [
        ("Luong Thanh Binh", "20200001", "DSAI-01", "CNTT"),
        ("Trinh Hoang Anh", "20200002", "DSAI-01", "CNTT"),
        ("Bao", "20200003", "DSAI-01", "CNTT")
    ]
    
    for name, mssv, dept, school in students:
        add_student_to_class(class_id, name, mssv, dept, school)
        
    print(f"--> Đã thêm {len(students)} sinh viên mẫu vào lớp {class_id}.")
    print("\n[SUCCESS] Khởi tạo dữ liệu hoàn tất! Bạn có thể chạy web_app.py ngay.")

if __name__ == "__main__":
    init_system()
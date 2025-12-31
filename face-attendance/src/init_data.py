import os
from pathlib import Path
from src.backend_manager import register_user, create_class, add_student_to_class

def init_system():
    
    base_dir = Path("data")
    (base_dir / "classes").mkdir(parents=True, exist_ok=True)
    (base_dir / "facebank").mkdir(parents=True, exist_ok=True) 
    
    register_user("admin", "00000")
    class_id = create_class("IT3320E", "Introduction to Deep Learning", 15)
    students = [
        ("Luong Thanh Binh", "20200001", "DSAI-01", "CNTT"),
        ("Trinh Hoang Anh", "20200002", "DSAI-01", "CNTT"),
        ("Bao", "20200003", "DSAI-01","DSAI-01", "CNTT")
    ]
    
    for name, mssv, dept, school in students:
        add_student_to_class(class_id, name, mssv, dept, school)
if __name__ == "__main__":
    init_system()

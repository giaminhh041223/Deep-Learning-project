import json
import os
import random
import string
import pandas as pd
from datetime import datetime
from pathlib import Path

# Cấu hình đường dẫn
# Giả sử file này nằm trong src/, thì parent.parent là thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"
CLASS_DIR = DATA_DIR / "classes"
USER_FILE = DATA_DIR / "users.json"

# Đảm bảo thư mục tồn tại
CLASS_DIR.mkdir(parents=True, exist_ok=True)

def generate_id(length=8):
    """Tạo ID ngẫu nhiên không trùng"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# --- QUẢN LÝ USER ---
def load_users():
    if not USER_FILE.exists(): return {}
    with open(USER_FILE, 'r', encoding='utf-8') as f: return json.load(f)

def save_users(users):
    with open(USER_FILE, 'w', encoding='utf-8') as f: json.dump(users, f, indent=4)

def register_user(username, password):
    users = load_users()
    if username in users: return False # Đã tồn tại
    users[username] = {
        "password": password, 
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_users(users)
    return True

def login_user(username, password):
    users = load_users()
    if username in users and users[username]["password"] == password:
        return True
    return False

# --- QUẢN LÝ LỚP HỌC ---
def create_class(class_code, lecturer, sessions):
    class_id = generate_id(8)
    # Đảm bảo ID không trùng (dù xác suất thấp)
    while (CLASS_DIR / f"{class_id}.json").exists():
        class_id = generate_id(8)
        
    class_data = {
        "class_id": class_id,
        "class_code": class_code,
        "lecturer": lecturer,
        "total_sessions": int(sessions),
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "students": [] # Danh sách sinh viên
    }
    save_class(class_id, class_data)
    return class_id

def save_class(class_id, data):
    file_path = CLASS_DIR / f"{class_id}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_class(class_id):
    file_path = CLASS_DIR / f"{class_id}.json"
    if not file_path.exists(): return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Hàm alias để tương thích với frontend gọi get_class_detail
def get_class_detail(class_id):
    return get_class(class_id)

def get_all_classes():
    classes = []
    # Quét tất cả file .json trong thư mục classes
    for file_path in CLASS_DIR.glob("*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Trả về format gọn nhẹ cho danh sách
            classes.append({
                "id": data.get("class_id"),
                "code": data.get("class_code"),
                "lecturer": data.get("lecturer"),
                "students": len(data.get("students", [])),
                "sessions": data.get("total_sessions")
            })
    return classes

# --- QUẢN LÝ SINH VIÊN & ĐIỂM SỐ ---
def calculate_score(absent_count):
    if absent_count == 0: return 1.0
    if 1 <= absent_count <= 2: return 0.0
    if absent_count == 3: return -0.5
    if absent_count == 4: return -1.0
    if absent_count == 5: return -1.5
    if absent_count >= 6: return -2.0
    return 0.0

def add_student_to_class(class_id, name, mssv, dept, school):
    class_data = get_class(class_id)
    if not class_data: return False
    
    # Kiểm tra trùng MSSV
    for s in class_data['students']:
        if s['mssv'] == mssv: return False

    new_student = {
        "student_id": generate_id(6),
        "name": name,
        "mssv": mssv,
        "department": dept,
        "school": school,
        "absent_count": 0,
        "attendance_score": 1.0,
        "history": [] 
    }
    
    class_data['students'].append(new_student)
    save_class(class_id, class_data)
    return True

# Alias function cho frontend (frontend gọi add_student)
def add_student(class_id, name, mssv, dept, school):
    return add_student_to_class(class_id, name, mssv, dept, school)

def import_from_excel(class_id, excel_path):
    """Nhập danh sách từ Excel (Cột: Name, MSSV, Dept, School)"""
    try:
        df = pd.read_excel(excel_path)
        count = 0
        for _, row in df.iterrows():
            name = str(row.get('Name', '')).strip()
            mssv = str(row.get('MSSV', '')).strip()
            dept = str(row.get('Dept', '')).strip()
            school = str(row.get('School', '')).strip()
            
            if name and mssv:
                if add_student_to_class(class_id, name, mssv, dept, school):
                    count += 1
        return count
    except Exception as e:
        print(f"[Error] Import Excel failed: {e}")
        return 0

# Alias cho frontend
def import_excel(class_id, excel_path):
    return import_from_excel(class_id, excel_path)

# --- LOGIC CẬP NHẬT ĐIỂM DANH ---
def update_attendance(class_id, mssv, checkin_time, checkout_time):
    """Tính toán thời gian và cập nhật trạng thái sau khi Checkout"""
    class_data = get_class(class_id)
    if not class_data: return False
    
    # Định dạng thời gian
    fmt = '%Y-%m-%d %H:%M:%S'
    try:
        t_in = datetime.strptime(checkin_time, fmt)
        t_out = datetime.strptime(checkout_time, fmt)
        # Tính thời lượng (phút)
        duration_min = (t_out - t_in).total_seconds() / 60
    except ValueError:
        return False # Lỗi format thời gian
    
    # Quy tắc: Thời lượng buổi học = số tiết * 50p.
    # Cần tham dự > 50% thời lượng buổi học.
    # Lấy mặc định 1 buổi = 2 tiết = 100p (nếu không có cấu hình chi tiết)
    session_duration = 2 * 50 
    required_min = session_duration * 0.5 
    
    status = "Present" if duration_min >= required_min else "Absent"
    
    found = False
    for stu in class_data['students']:
        if stu['mssv'] == mssv:
            found = True
            # Nếu vắng, cập nhật điểm trừ
            if status == "Absent":
                stu['absent_count'] += 1
                stu['attendance_score'] = calculate_score(stu['absent_count'])
            
            # Lưu lịch sử
            record = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "checkin": checkin_time,
                "checkout": checkout_time,
                "duration": round(duration_min, 1),
                "status": status
            }
            stu['history'].append(record)
            break
            
    if found:
        save_class(class_id, class_data)
        return True
    return False

# Alias cho frontend (frontend gọi process_checkout)
def process_checkout(class_id, mssv, checkin_time, checkout_time, session_duration_minutes):
    # Logic tương tự update_attendance nhưng trả về nhiều info hơn
    class_data = get_class(class_id)
    if not class_data: return False, "Error", 0
    
    fmt = '%Y-%m-%d %H:%M:%S'
    try:
        t_in = datetime.strptime(checkin_time, fmt)
        t_out = datetime.strptime(checkout_time, fmt)
        duration_min = (t_out - t_in).total_seconds() / 60
    except: return False, "Time Error", 0

    threshold = session_duration_minutes * 0.5
    status = "Present" if duration_min >= threshold else "Absent"
    
    found = False
    for s in class_data['students']:
        if s['mssv'] == mssv:
            found = True
            s['history'].append({
                "date": datetime.now().strftime('%Y-%m-%d'), 
                "checkin": checkin_time,
                "checkout": checkout_time, 
                "duration": round(duration_min, 1), 
                "status": status
            })
            # Recalculate score based on total history
            total_absent = sum(1 for h in s['history'] if h['status'] == 'Absent')
            s['absent_count'] = total_absent
            s['attendance_score'] = calculate_score(total_absent)
            break
            
    if found:
        save_class(class_id, class_data)
        return True, status, round(duration_min, 1)
    return False, "Not Found", 0
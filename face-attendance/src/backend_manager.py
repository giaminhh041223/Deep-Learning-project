import json
import os
import random
import string
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"
CLASS_DIR = DATA_DIR / "classes"
USER_FILE = DATA_DIR / "users.json"

CLASS_DIR.mkdir(parents=True, exist_ok=True)

def generate_id(length=8):
    """Create random ID"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def load_users():
    if not USER_FILE.exists(): return {}
    with open(USER_FILE, 'r', encoding='utf-8') as f: return json.load(f)

def save_users(users):
    with open(USER_FILE, 'w', encoding='utf-8') as f: json.dump(users, f, indent=4)

def register_user(username, password):
    users = load_users()
    if username in users: return False 
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

def create_class(class_code, lecturer, sessions):
    class_id = generate_id(8)
    while (CLASS_DIR / f"{class_id}.json").exists():
        class_id = generate_id(8)
        
    class_data = {
        "class_id": class_id,
        "class_code": class_code,
        "lecturer": lecturer,
        "total_sessions": int(sessions),
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "students": [] 
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

def get_class_detail(class_id):
    return get_class(class_id)

def get_all_classes():
    classes = []
    for file_path in CLASS_DIR.glob("*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            classes.append({
                "id": data.get("class_id"),
                "code": data.get("class_code"),
                "lecturer": data.get("lecturer"),
                "students": len(data.get("students", [])),
                "sessions": data.get("total_sessions")
            })
    return classes

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

def add_student(class_id, name, mssv, dept, school):
    return add_student_to_class(class_id, name, mssv, dept, school)

def import_from_excel(class_id, excel_path):
    """Inport from Excel (Name, MSSV, Dept, School)"""
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
def import_excel(class_id, excel_path):
    return import_from_excel(class_id, excel_path)

def update_attendance(class_id, mssv, checkin_time, checkout_time):
    """Calculate total time and update status after Checkout"""
    class_data = get_class(class_id)
    if not class_data: return False
    fmt = '%Y-%m-%d %H:%M:%S'
    try:
        t_in = datetime.strptime(checkin_time, fmt)
        t_out = datetime.strptime(checkout_time, fmt)
        # minutes
        duration_min = (t_out - t_in).total_seconds() / 60
    except ValueError:
        return False

    session_duration = 2 * 50 
    required_min = session_duration * 0.5 
    
    status = "Present" if duration_min >= required_min else "Absent"
    
    found = False
    for stu in class_data['students']:
        if stu['mssv'] == mssv:
            found = True
            if status == "Absent":
                stu['absent_count'] += 1
                stu['attendance_score'] = calculate_score(stu['absent_count'])
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

def process_checkout(class_id, mssv, checkin_time, checkout_time, session_duration_minutes):
    class_data = get_class(class_id)
    if not class_data: return False, "Error", 0
    
    fmt = '%Y-%m-%d %H:%M:%S'
    try:
        t_in = datetime.strptime(checkin_time, fmt)
        t_out = datetime.strptime(checkout_time, fmt)
        duration_min = (t_out - t_in).total_seconds() / 60
    except: return False, "Time Error", 0

    checkpoint = session_duration_minutes * 0.5
    status = "Present" if duration_min >= checkpoint else "Absent"
    
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
            total_absent = sum(1 for h in s['history'] if h['status'] == 'Absent')
            s['absent_count'] = total_absent
            s['attendance_score'] = calculate_score(total_absent)
            break
            
    if found:
        save_class(class_id, class_data)
        return True, status, round(duration_min, 1)
    return False, "Not Found", 0

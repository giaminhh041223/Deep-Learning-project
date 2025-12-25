import os
import cv2
import time
from datetime import datetime
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from werkzeug.utils import secure_filename

# --- CẤU HÌNH DLL CUDA (WINDOWS) ---
if os.name == 'nt':
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        os.add_dll_directory(project_root)
        print(f"[INFO] Đã thêm DLL directory: {project_root}")
    except Exception as e:
        print(f"[WARN] Không thể thêm DLL directory: {e}")

# Import Modules
from src.config import CFG
from src.detection import Detector
from src.alignment import Aligner
from src.embedding import Embedder
from src.matching import FaceBank
from src.draw import draw_bbox_landmarks, draw_label
import src.backend_manager as db

# --- FLASK SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
if not os.path.exists(template_dir):
    template_dir = os.path.join(os.path.dirname(current_dir), 'src', 'templates')

app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'face_attendance_secret_key_super_secure' 

# --- AI INIT ---
print("[INFO] Loading AI Models...")
try:
    detector = Detector()
    aligner = Aligner(112)
    embedder = Embedder()
    facebank = FaceBank()
    print("[INFO] AI Ready!")
except Exception as e:
    print(f"[ERROR] AI Init Failed: {e}")

# --- GLOBAL STATE ---
# current_class_id đã bị xóa để tránh lỗi logic
live_sessions = {} 
current_detected_student = {"name": "Unknown", "mssv": "---", "info": {}}

# --- HELPER ---
def get_session_duration(class_id):
    return 90 

# --- ROUTES ---
@app.route('/')
def index():
    if 'user' in session: return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pw = request.form.get('password')
        if db.login_user(user, pw):
            session['user'] = user
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid Username or Password")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = request.form.get('username')
        pw = request.form.get('password')
        if db.register_user(user, pw):
            return redirect(url_for('login'))
        return render_template('register.html', error="Username already exists")
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('login'))
    classes = db.get_all_classes()
    return render_template('dashboard.html', user=session['user'], classes=classes)

@app.route('/students')
def students_page():
    if 'user' not in session: return redirect(url_for('login'))
    all_students = []
    classes = db.get_all_classes()
    for cls_meta in classes:
        cls_data = db.get_class_detail(cls_meta['id'])
        if cls_data and 'students' in cls_data:
            for s in cls_data['students']:
                s_copy = s.copy()
                s_copy['class_code'] = cls_data['class_code']
                s_copy['class_id'] = cls_data['class_id']
                all_students.append(s_copy)
    return render_template('students.html', user=session['user'], students=all_students)

@app.route('/reports')
def reports_page():
    if 'user' not in session: return redirect(url_for('login'))
    classes = db.get_all_classes()
    summary = []
    total_students = 0
    total_classes = len(classes)
    for cls_meta in classes:
        cls_data = db.get_class_detail(cls_meta['id'])
        if cls_data:
            s_count = len(cls_data.get('students', []))
            total_students += s_count
            passing = sum(1 for s in cls_data['students'] if s['attendance_score'] > 0)
            rate = (passing / s_count * 100) if s_count > 0 else 0
            summary.append({
                "name": cls_data['class_code'],
                "students": s_count,
                "rate": round(rate, 1)
            })
    return render_template('reports.html', user=session['user'], summary=summary, total_students=total_students, total_classes=total_classes)

@app.route('/create_class', methods=['POST'])
def create_class():
    if 'user' not in session: return redirect(url_for('login'))
    code = request.form.get('code')
    lecturer = request.form.get('lecturer')
    sessions = request.form.get('sessions')
    class_id = db.create_class(code, lecturer, sessions)
    
    file = request.files.get('excel_file')
    if file and file.filename:
        filename = secure_filename(file.filename)
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, 'data')
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        path = os.path.join(data_dir, filename)
        file.save(path)
        db.import_excel(class_id, path)
        try: os.remove(path)
        except: pass
    return redirect(url_for('dashboard'))

@app.route('/attendance/<class_id>')
def attendance(class_id):
    if 'user' not in session: return redirect(url_for('login'))
    # Không dùng global current_class_id nữa
    class_info = db.get_class_detail(class_id)
    return render_template('attendance.html', user=session['user'], class_info=class_info)

@app.route('/class_detail/<class_id>')
def class_detail(class_id):
    if 'user' not in session: return redirect(url_for('login'))
    info = db.get_class_detail(class_id)
    return render_template('class_detail.html', user=session['user'], class_info=info)

# --- API ---
@app.route('/api/detect_info')
def api_detect_info():
    return jsonify(current_detected_student)

@app.route('/api/mark', methods=['POST'])
def api_mark():
    global live_sessions
    data = request.json
    action = data.get('action')
    mssv = data.get('mssv')
    name = data.get('name')
    class_id = data.get('class_id') # Nhận class_id từ frontend

    if not class_id:
        return jsonify({"success": False, "msg": "Class ID missing!"})

    if not mssv or mssv == "---":
        return jsonify({"success": False, "msg": "No student detected!"})

    # Kiểm tra sinh viên có trong lớp không
    class_info = db.get_class_detail(class_id)
    is_in_class = False
    if class_info:
        for s in class_info.get('students', []):
            if s['mssv'] == mssv:
                is_in_class = True
                break
    
    if not is_in_class:
        return jsonify({"success": False, "msg": f"Student {name} is not in this class!"})

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if action == 'in':
        if mssv in live_sessions:
            return jsonify({"success": False, "msg": f"{name} already Checked-in!"})
        live_sessions[mssv] = {"in": now_str, "name": name}
        return jsonify({"success": True, "msg": f"Checked IN: {name} at {now_str}"})

    elif action == 'out':
        if mssv not in live_sessions:
            return jsonify({"success": False, "msg": f"{name} has NOT Checked-in yet!"})
        
        checkin_time = live_sessions[mssv]['in']
        checkout_time = now_str
        session_mins = get_session_duration(class_id)
        ok, status, dur = db.process_checkout(class_id, mssv, checkin_time, checkout_time, session_mins)
        
        if ok:
            del live_sessions[mssv]
            return jsonify({"success": True, "msg": f"Checked OUT: {name}. Duration: {dur}m. Status: {status}"})
        else:
            return jsonify({"success": False, "msg": "Error updating database."})

    return jsonify({"success": False, "msg": "Invalid action"})

@app.route('/api/register_new', methods=['POST'])
def api_register_new():
    data = request.json
    name = data.get('name')
    mssv = data.get('mssv')
    class_id = data.get('class_id') # Nhận class_id từ frontend

    if not class_id:
         return jsonify({"success": False, "msg": "Class ID missing!"})

    # 1. Kiểm tra Global DB (FaceBank)
    student_in_global = False
    for record in facebank.meta:
        if record.get('mssv') == mssv:
            student_in_global = True
            break
    
    # 2. Thêm vào Lớp học
    added = db.add_student(class_id, name, mssv, "General", "HUST")
    
    msg = f"Added {name} to class." if added else f"{name} already in class."
    
    # 3. Phản hồi
    if student_in_global:
        return jsonify({"success": True, "msg": f"{msg} (Face data exists)", "require_capture": False})
    else:
        return jsonify({"success": True, "msg": f"{msg} (New face - Capture needed)", "require_capture": True})

# --- STREAM ---
def generate_frames():
    global current_detected_student
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened(): camera = cv2.VideoCapture(0)
    
    frame_count = 0
    while True:
        success, frame = camera.read()
        if not success: break
        
        img = frame.copy()
        if frame_count % 3 == 0:
            dets = detector.detect(img, min_face=CFG.min_face)
            best_face = None
            max_area = 0
            for d in dets:
                bbox = d['bbox']
                area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                if area > max_area:
                    max_area = area
                    best_face = d
            
            if best_face:
                kps = best_face['kps']
                crop = aligner.align(img, kps)
                emb = embedder.embed(crop)
                label, score, info = facebank.match(emb, threshold=CFG.threshold)
                
                real_name = info.get('name', 'Unknown') if info else label.split('|')[0]
                real_mssv = info.get('mssv', '---') if info else '---'
                
                current_detected_student = {"name": real_name, "mssv": real_mssv, "info": info}
                
                color = (0, 255, 0) if real_name != "Unknown" else (0, 0, 255)
                draw_bbox_landmarks(img, best_face['bbox'], kps, color)
                draw_label(img, best_face['bbox'], real_name, color)
            else:
                current_detected_student = {"name": "Unknown", "mssv": "---"}
        
        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

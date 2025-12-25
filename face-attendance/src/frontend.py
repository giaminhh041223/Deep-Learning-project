import os
import cv2
import time
from datetime import datetime
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from werkzeug.utils import secure_filename

# --- CẤU HÌNH FIX LỖI GPU WINDOWS ---
if os.name == 'nt':
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        os.add_dll_directory(project_root)
    except Exception as e:
        pass

# Import Modules
from src.config import CFG
from src.detection import Detector
from src.alignment import Aligner
from src.embedding import Embedder
from src.matching import FaceBank
from src.draw import draw_bbox_landmarks, draw_label
import src.backend_manager as db

# --- CẤU HÌNH FLASK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
if not os.path.exists(template_dir):
    template_dir = os.path.join(os.path.dirname(current_dir), 'src', 'templates')

app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'super_secret_key'

# --- KHỞI TẠO AI ---
print("[INFO] Loading AI Models...")
detector = Detector()
aligner = Aligner(112)
embedder = Embedder()
facebank = FaceBank() # Load dữ liệu khuôn mặt từ disk vào RAM
print("[INFO] AI Ready!")

# --- BIẾN TOÀN CỤC ---
current_class_id = None
live_sessions = {}
current_detected_student = {"name": "Unknown", "mssv": "---", "info": {}}

# --- TRẠNG THÁI ĐĂNG KÝ (REGISTRATION STATE) ---
reg_state = {
    "active": False,      # Đang trong chế độ chụp ảnh hay không
    "name": "",
    "mssv": "",
    "class_id": "",
    "count": 0,           # Số ảnh đã chụp
    "target": 5,          # Mục tiêu số ảnh cần chụp
    "embeddings": [],     # Lưu tạm các vector đặc trưng
    "message": ""
}

def get_session_duration(class_id):
    return 90

# ================= ROUTES =================

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
        return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

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
            summary.append({"name": cls_data['class_code'], "students": s_count, "rate": round(rate, 1)})
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
        path = os.path.join(os.path.dirname(current_dir), 'data', filename)
        file.save(path)
        db.import_excel(class_id, path)
        try: os.remove(path)
        except: pass
    return redirect(url_for('dashboard'))

@app.route('/attendance/<class_id>')
def attendance(class_id):
    if 'user' not in session: return redirect(url_for('login'))
    # Không dùng global current_class_id nữa để tránh conflict
    class_info = db.get_class_detail(class_id)
    return render_template('attendance.html', user=session['user'], class_info=class_info)

@app.route('/class_detail/<class_id>')
def class_detail(class_id):
    if 'user' not in session: return redirect(url_for('login'))
    info = db.get_class_detail(class_id)
    return render_template('class_detail.html', user=session['user'], class_info=info)

# ================= API ENDPOINTS =================

@app.route('/api/detect_info')
def api_detect_info():
    """Trả về thông tin người đang detect và trạng thái đăng ký"""
    return jsonify({
        "name": current_detected_student["name"],
        "mssv": current_detected_student["mssv"],
        "reg_status": {
            "active": reg_state["active"],
            "count": reg_state["count"],
            "target": reg_state["target"],
            "message": reg_state["message"]
        }
    })

@app.route('/api/mark', methods=['POST'])
def api_mark():
    global live_sessions
    data = request.json
    action = data.get('action')
    mssv = data.get('mssv')
    name = data.get('name')
    class_id = data.get('class_id')

    if not mssv or mssv == "---" or name == "Unknown":
        return jsonify({"success": False, "msg": "No student detected!"})

    # Validate Class
    class_info = db.get_class_detail(class_id)
    is_in_class = False
    if class_info:
        for s in class_info.get('students', []):
            if s['mssv'] == mssv:
                is_in_class = True
                break
    
    if not is_in_class:
        return jsonify({"success": False, "msg": f"Student {name} is NOT in this class!"})

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if action == 'in':
        if mssv in live_sessions:
            return jsonify({"success": False, "msg": "Already Checked-in!"})
        live_sessions[mssv] = {"in": now_str}
        return jsonify({"success": True, "msg": f"Checked IN at {now_str}"})

    elif action == 'out':
        if mssv not in live_sessions:
            return jsonify({"success": False, "msg": "Not Checked-in yet!"})
        
        checkin_time = live_sessions[mssv]['in']
        # Tính toán và lưu DB
        ok, status, dur = db.process_checkout(class_id, mssv, checkin_time, now_str, get_session_duration(class_id))
        del live_sessions[mssv]
        
        return jsonify({"success": True, "msg": f"Checked OUT. Duration: {dur}m. Status: {status}"})

    return jsonify({"success": False, "msg": "Invalid action"})

@app.route('/api/register_new', methods=['POST'])
def api_register_new():
    """
    API xử lý đăng ký:
    1. Kiểm tra trong Global DB (FaceBank).
    2. Nếu chưa có -> Bật chế độ chụp ảnh (Capture Mode).
    3. Nếu có rồi -> Chỉ thêm vào danh sách lớp.
    """
    data = request.json
    name = data.get('name')
    mssv = data.get('mssv')
    class_id = data.get('class_id')
    
    if not name or not mssv: return jsonify({"success": False, "msg": "Missing Info"})

    # 1. Kiểm tra xem đã có Face Data chưa
    has_face_data = False
    for record in facebank.meta:
        if record.get('mssv') == mssv:
            has_face_data = True
            break
            
    # 2. Luôn thêm vào danh sách lớp học (Class DB)
    # Hàm này trong backend_manager đã check trùng
    db.add_student(class_id, name, mssv, "General", "HUST")
    
    if has_face_data:
        return jsonify({"success": True, "msg": f"Added {name} to class list (Face data exists).", "require_capture": False})
    else:
        # 3. Kích hoạt chế độ chụp ảnh
        reg_state["active"] = True
        reg_state["name"] = name
        reg_state["mssv"] = mssv
        reg_state["class_id"] = class_id
        reg_state["count"] = 0
        reg_state["embeddings"] = []
        reg_state["message"] = "Starting capture..."
        
        return jsonify({"success": True, "msg": "Starting face capture...", "require_capture": True})

# ================= VIDEO STREAM LOOP =================

def generate_frames():
    global current_detected_student, reg_state
    
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened(): camera = cv2.VideoCapture(0)
    
    frame_count = 0
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        img = frame.copy()
        
        # --- CHẾ ĐỘ ĐĂNG KÝ (CAPTURE MODE) ---
        if reg_state["active"]:
            # Hiển thị hướng dẫn
            cv2.putText(img, f"CAPTURING: {reg_state['count']}/{reg_state['target']}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.putText(img, "Please rotate your face slightly", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Phát hiện khuôn mặt để chụp
            faces = detector.detect(img, min_face=CFG.min_face)
            
            if faces:
                # Lấy mặt lớn nhất
                best_face = max(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
                bbox = best_face['bbox']
                kps = best_face['kps']
                
                # Vẽ khung màu cam
                draw_bbox_landmarks(img, bbox, kps, color=(0, 165, 255))
                
                # Chụp mỗi 5 frame một lần để tránh ảnh giống nhau quá
                if frame_count % 5 == 0 and reg_state["count"] < reg_state["target"]:
                    # Crop & Align
                    crop = aligner.align(img, kps)
                    # Extract Embedding
                    emb = embedder.embed(crop)
                    
                    reg_state["embeddings"].append(emb)
                    reg_state["count"] += 1
                    
                    # --- KHI CHỤP ĐỦ ẢNH ---
                    if reg_state["count"] >= reg_state["target"]:
                        reg_state["message"] = "Processing..."
                        
                        # 1. Tính trung bình vector
                        mean_feat = np.mean(np.stack(reg_state["embeddings"]), axis=0)
                        # Chuẩn hóa vector (Quan trọng!)
                        mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-9)
                        
                        # 2. Thêm vào FaceBank (RAM + Disk)
                        # Hàm add sẽ cập nhật self.embeddings trong RAM -> Nhận diện được ngay
                        facebank.add(mean_feat, reg_state["name"], "Unknown", reg_state["mssv"])
                        facebank.save() # Lưu xuống đĩa (embeddings.npy, meta.json)
                        
                        print(f"[SUCCESS] Registered new face: {reg_state['name']}")
                        
                        # 3. Tắt chế độ đăng ký
                        reg_state["active"] = False
                        reg_state["message"] = "Đăng ký thành công!"

        # --- CHẾ ĐỘ ĐIỂM DANH (NORMAL MODE) ---
        else:
            # Nhận diện mỗi 3 frame
            if frame_count % 3 == 0:
                dets = detector.detect(img, min_face=CFG.min_face)
                
                # Tìm mặt lớn nhất
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
                    
                    # Nhận diện
                    label, score, info = facebank.match(emb, threshold=CFG.threshold)
                    
                    real_name = info.get('name', 'Unknown') if info else label.split('|')[0]
                    real_mssv = info.get('mssv', '---') if info else '---'
                    
                    current_detected_student = {"name": real_name, "mssv": real_mssv, "info": info}
                    
                    color = (0, 255, 0) if real_name != "Unknown" else (0, 0, 255)
                    draw_bbox_landmarks(img, best_face['bbox'], kps, color)
                    draw_label(img, best_face['bbox'], real_name, color)
                else:
                    current_detected_student = {"name": "Unknown", "mssv": "---", "info": {}}

        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

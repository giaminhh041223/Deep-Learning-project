import os
import cv2
import time
import datetime
import numpy as np
import threading
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request
from src.config import CFG
from src.detection import Detector
from src.alignment import Aligner
from src.embedding import Embedder
from src.matching import FaceBank
from src.draw import draw_bbox_landmarks, draw_label

# --- CẤU HÌNH FLASK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
if not os.path.exists(template_dir):
    template_dir = os.path.join(os.path.dirname(current_dir), 'templates')

app = Flask(__name__, template_folder=template_dir)

# --- KHỞI TẠO MODELS ---
print("[INFO] Đang tải các mô hình AI...")
detector = Detector()
aligner = Aligner(112)
embedder = Embedder()
facebank = FaceBank()
print("[INFO] Hệ thống sẵn sàng!")

# --- DB ĐIỂM DANH & TRẠNG THÁI ---
attendance_db = {}
current_user_detected = {"name": "Chưa xác định", "info": {}}

# --- BIẾN TRẠNG THÁI CHO QUÁ TRÌNH ĐĂNG KÝ ---
registration_state = {
    "is_active": False,
    "step": 0,          
    "person_name": "",
    "person_mssv": "",
    "person_class": "",
    "captured_count": 0,
    "target_count": 5,  
    "message": "",
    "captured_embeddings": [] 
}

def format_duration(seconds):
    if not seconds: return "0h 0m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

def process_manual_attendance(action_type):
    """Xử lý khi bấm nút Vào Lớp/Ra Về"""
    global attendance_db, current_user_detected
    
    name = current_user_detected["name"]
    info = current_user_detected["info"]
    now = time.time()

    if name == "Chưa xác định" or name == "Unknown":
        return False, "Không nhận diện được khuôn mặt! Vui lòng đứng chính diện."

    if name not in attendance_db:
        attendance_db[name] = {
            "name": info.get('name', name),
            "mssv": info.get('mssv', '---'),
            "class_name": info.get('class', '---'),
            "status": "N/A", 
            "checkin_time": None, 
            "checkout_time": None, 
            "total_work": 0,
            "history": []
        }

    user = attendance_db[name]
    msg = ""
    success = True
    time_str = time.strftime('%H:%M:%S')

    # NÚT VÀO LỚP
    if action_type == "checkin":
        if user["status"] == "CHECKED_IN":
            success = False
            msg = f"Sinh viên {name} đã điểm danh rồi!"
        else:
            user["status"] = "CHECKED_IN"
            user["checkin_time"] = now
            msg = f"Xin chào {name}! Đã vào lớp lúc {time_str}."
            user["history"].insert(0, {"action": "Vào Lớp", "time": time_str})

    # NÚT RA VỀ
    elif action_type == "checkout":
        if user["status"] != "CHECKED_IN":
            success = False
            msg = f"Sinh viên {name} chưa vào lớp, không thể ra về!"
        else:
            user["status"] = "CHECKED_OUT"
            user["checkout_time"] = now
            
            # Tính giờ học
            session = now - user["checkin_time"]
            user["total_work"] += session
            total_str = format_duration(user["total_work"])
            
            msg = f"Tạm biệt {name}! Thời gian học: {total_str}."
            user["history"].insert(0, {"action": "Ra Về", "time": time_str, "duration": total_str})

    return success, msg

def generate_frames():
    global current_user_detected, registration_state
    
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened(): camera = cv2.VideoCapture(0)

    frame_count = 0
    detect_interval = 3
    last_capture_time = 0
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        img = frame.copy()

        # --- LOGIC ĐĂNG KÝ ---
        if registration_state["is_active"]:
            cv2.putText(img, f"MODE: DANG KY ({registration_state['captured_count']}/{registration_state['target_count']})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            now = time.time()
            dets = detector.detect(img, min_face=CFG.min_face)
            
            if dets:
                best_face = max(dets, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]))
                bbox = best_face['bbox']
                kps = best_face['kps']
                
                draw_bbox_landmarks(img, bbox, kps, color=(0, 165, 255))
                
                if registration_state["captured_count"] < registration_state["target_count"]:
                    if now - last_capture_time > 0.8: 
                        crop = aligner.align(img, kps)
                        emb = embedder.embed(crop)
                        
                        registration_state["captured_embeddings"].append(emb)
                        registration_state["captured_count"] += 1
                        last_capture_time = now
                        registration_state["message"] = "Đang chụp... Giữ nguyên."
                else:
                    # Giai đoạn lưu dữ liệu
                    if registration_state["step"] != 2: # Đảm bảo chỉ chạy 1 lần
                        registration_state["step"] = 2
                        registration_state["message"] = "Đang lưu dữ liệu..."
                        
                        try:
                            feats = np.stack(registration_state["captured_embeddings"], axis=0)
                            mean_feat = np.mean(feats, axis=0)
                            mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-9)
                            
                            # Thêm vào FaceBank
                            facebank.add(mean_feat, 
                                         name=registration_state["person_name"],
                                         cls=registration_state["person_class"],
                                         mssv=registration_state["person_mssv"])
                            
                            # Lưu file ngay lập tức
                            facebank.save()
                            print(f"[INFO] Đã lưu sinh viên mới: {registration_state['person_name']} vào {facebank.dir}")
                            
                            registration_state["message"] = "Đăng ký thành công!"
                        except Exception as e:
                            print(f"[ERROR] Lưu thất bại: {e}")
                            registration_state["message"] = "Lỗi khi lưu!"

                    # Tự động tắt sau 2 giây
                    if now - last_capture_time > 2.0:
                        registration_state["is_active"] = False
                        
            else:
                cv2.putText(img, "Khong thay khuon mat!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- LOGIC ĐIỂM DANH ---
        else:
            if frame_count % detect_interval == 0:
                dets = detector.detect(img, min_face=CFG.min_face)
                found_someone = False
                
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
                    
                    label_str, score, info = facebank.match(emb, threshold=CFG.threshold)
                    real_name = info.get('name', 'Unknown') if info else label_str.split('|')[0].strip()
                    
                    if real_name != "Unknown":
                        current_user_detected = {"name": real_name, "info": info}
                        found_someone = True
                    
                    color = (0, 255, 0) if real_name != "Unknown" else (0, 0, 255)
                    draw_bbox_landmarks(img, best_face['bbox'], kps, color=color)
                    
                    disp = f"{real_name}"
                    if real_name in attendance_db:
                        stat = attendance_db[real_name]["status"]
                        if stat == "CHECKED_IN": disp += " [Lop]"
                    
                    draw_label(img, best_face['bbox'], disp, color=color)
                
                if not found_someone:
                    current_user_detected = {"name": "Chưa xác định", "info": {}}

        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/action', methods=['POST'])
def handle_action():
    data = request.json
    action = data.get('action') 
    success, msg = process_manual_attendance(action)
    return jsonify({"success": success, "message": msg})

@app.route('/api/history', methods=['GET'])
def get_history():
    name = request.args.get('name')
    if name and name in attendance_db:
        return jsonify(attendance_db[name]["history"])
    return jsonify([])

@app.route('/api/stats')
def get_stats():
    data = []
    sorted_users = sorted(attendance_db.values(), key=lambda x: max(x['checkin_time'] or 0, x['checkout_time'] or 0), reverse=True)
    
    for u in sorted_users:
        checkin = time.strftime('%H:%M:%S', time.localtime(u['checkin_time'])) if u['checkin_time'] else "--:--"
        checkout = time.strftime('%H:%M:%S', time.localtime(u['checkout_time'])) if u['checkout_time'] else "--:--"
        total = format_duration(u['total_work'])
        
        running_time = ""
        if u['status'] == "CHECKED_IN":
            elapsed = time.time() - u['checkin_time']
            running_time = format_duration(elapsed)

        data.append({
            "mssv": u['mssv'],
            "name": u['name'],
            "class_name": u['class_name'],
            "status": u['status'],
            "checkin": checkin,
            "checkout": checkout,
            "total": total,
            "running": running_time
        })
        
    return jsonify({
        "users": data, 
        "current_detect": current_user_detected["name"],
        "reg_status": {
            "active": registration_state["is_active"],
            "count": registration_state["captured_count"],
            "total": registration_state["target_count"],
            "message": registration_state["message"]
        }
    })

@app.route('/api/register', methods=['POST'])
def start_register():
    data = request.json
    name = data.get('name')
    mssv = data.get('mssv')
    cls = data.get('class')
    
    if not name or not mssv:
        return jsonify({"success": False, "message": "Thiếu thông tin bắt buộc!"})
        
    registration_state["is_active"] = True
    registration_state["step"] = 1 # Bắt đầu chụp
    registration_state["person_name"] = name
    registration_state["person_mssv"] = mssv
    registration_state["person_class"] = cls
    registration_state["captured_count"] = 0
    registration_state["captured_embeddings"] = []
    registration_state["message"] = "Bắt đầu chụp..."
    
    return jsonify({"success": True, "message": "Đã chuyển sang chế độ đăng ký"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
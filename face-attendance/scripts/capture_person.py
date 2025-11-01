import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import unicodedata as ud
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

def N(s: str) -> str:
    return ud.normalize("NFC", s or "").strip()

def _default_providers():
    avail = set(ort.get_available_providers())
    order = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    return [p for p in order if p in avail]

def open_cam(src: str|int, w: int, h: int):
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/source: {src}")
    if w > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    if h > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def expand_bbox(bbox, scale=1.2, img_w=None, img_h=None):
    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = (x2 - x1) * scale
    h  = (y2 - y1) * scale
    nx1 = int(max(0, cx - w/2))
    ny1 = int(max(0, cy - h/2))
    nx2 = int(min(img_w-1, cx + w/2)) if img_w else int(cx + w/2)
    ny2 = int(min(img_h-1, cy + h/2)) if img_h else int(cy + h/2)
    return nx1, ny1, nx2, ny2

def main():
    ap = argparse.ArgumentParser(description="Capture face crops (and aligned faces) into data/people/<name>/ at a time-based rate")
    ap.add_argument('--people_dir', default='data/people')
    ap.add_argument('--num', type=int, default=20, help='Số ảnh cần lưu')
    ap.add_argument('--duration', type=float, default=5.0, help='Thời gian (giây) để chụp đủ num ảnh (ví dụ 5s cho 20 ảnh)')
    ap.add_argument('--source', default='0')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--det_size', type=int, default=640)
    ap.add_argument('--min_face', type=int, default=80)
    ap.add_argument('--expand', type=float, default=1.3)
    ap.add_argument('--start_delay', type=float, default=0.0, help='Đếm lùi trước khi bắt đầu (giây), ví dụ 2.0')
    args = ap.parse_args()

    # Nhập tên qua input
    person_name = input("Nhập tên người cần chụp: ").strip()
    person_name = N(person_name)
    if not person_name:
        print("[ERROR] Tên trống. Thoát.")
        return

    people_dir = Path(args.people_dir)
    person_dir = people_dir / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    providers = _default_providers()
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    cap = open_cam(args.source, args.width, args.height)

    # Nhịp thời gian: mỗi interval giây chụp 1 ảnh
    num = max(1, args.num)
    duration = max(0.1, args.duration)
    interval = duration / num  # ví dụ 5/20 = 0.25s/ảnh

    count = 0
    print(f"[INFO] Bắt đầu chụp cho: {person_name}")
    if args.start_delay > 0:
        print(f"[INFO] Sẽ bắt đầu sau {args.start_delay:.1f}s...")
        time.sleep(args.start_delay)

    start_time = time.time()
    next_capture_t = start_time  # chụp ngay ảnh đầu tiên
    last_msg_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Không đọc được khung hình.")
            break

        img = frame.copy()
        faces = app.get(img)

        # Chọn mặt lớn nhất
        target = None
        max_area = 0
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            if (x2-x1) < args.min_face or (y2-y1) < args.min_face:
                continue
            area = (x2-x1)*(y2-y1)
            if area > max_area:
                max_area = area
                target = f

        # HUD
        now = time.time()
        if target is not None:
            x1, y1, x2, y2 = target.bbox.astype(int)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        # Thông tin tiến độ + đồng hồ
        elapsed = now - start_time
        txt = f"{person_name}  {count}/{num}  t={elapsed:0.1f}s  interval={interval:0.2f}s"
        cv2.putText(img, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if target is None and (now - last_msg_t) > 0.5:
            cv2.putText(img, "Khong thay mat - di vao khung hinh", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
            last_msg_t = now

        # Điều kiện chụp theo thời gian
        if target is not None and now >= next_capture_t and count < num:
            idx = count + 1

            # 1) Lưu crop thô
            h, w = frame.shape[:2]
            cx1, cy1, cx2, cy2 = expand_bbox(target.bbox, scale=args.expand, img_w=w, img_h=h)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                cv2.imwrite(str(person_dir / f"img_{idx:03d}.jpg"), crop)

            # 2) Lưu ảnh align 112x112
            aligned = norm_crop(frame, target.kps, image_size=112)
            cv2.imwrite(str(person_dir / f"align_{idx:03d}.jpg"), aligned)

            count += 1
            next_capture_t += interval  # hẹn lần chụp tiếp theo

        # Hiển thị
        cv2.imshow(f"Capture — {person_name}", img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            print("[INFO] Quit by user.")
            break

        # Dừng khi đủ num ảnh
        if count >= num:
            print(f"[OK] Đã lưu đủ {count} ảnh trong ~{(time.time()-start_time):.2f}s vào: {person_dir}")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[NOTE] Cập nhật CSV nếu cần (name,class,mssv) rồi build facebank:")
    print("  python -m scripts.register_faces --people_dir data/people --meta_csv data/people/meta.csv")

if __name__ == "__main__":
    main()

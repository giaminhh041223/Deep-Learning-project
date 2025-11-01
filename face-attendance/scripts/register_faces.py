import argparse
from pathlib import Path
import csv
import json
import cv2
import numpy as np
import unicodedata as ud

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from src.matching import FaceBank

def N(x: str) -> str:
    # Chuẩn hoá Unicode để tránh lệch tên có dấu trên macOS (NFD vs NFC)
    return ud.normalize("NFC", x or "")

def read_meta_csv(csv_path: Path):
    meta = {}
    if csv_path and csv_path.exists():
        with csv_path.open('r', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                name = N(row.get('name', '').strip())
                if not name:
                    continue
                meta[name] = {
                    "class": N(row.get('class','').strip()),
                    "mssv": N(row.get('mssv','').strip())
                }
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--people_dir', type=str, required=True, help='data/people')
    ap.add_argument('--meta_csv', type=str, default='')
    ap.add_argument('--min_images', type=int, default=3)
    args = ap.parse_args()

    people_dir = Path(args.people_dir)
    meta_csv = Path(args.meta_csv) if args.meta_csv else None

    # Load detector/recognizer từ buffalo_l
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640,640))

    # Khởi tạo facebank mới (xoá dữ liệu cũ nếu có)
    facebank = FaceBank()
    facebank.embeddings = np.zeros((0, 512), dtype=np.float32)
    facebank.meta = []
    facebank.save()  # ghi rỗng để tránh đọc lại dữ liệu cũ

    meta_map = read_meta_csv(meta_csv) if meta_csv else {}

    # Liệt kê thư mục người dùng, chuẩn hoá tên và sort cố định theo tên NFC
    person_dirs = []
    for p in people_dir.iterdir():
        if p.is_dir():
            person_dirs.append((N(p.name), p))
    person_dirs.sort(key=lambda x: x[0])  # sort theo tên NFC tăng dần

    for norm_name, person_dir in person_dirs:
        images = sorted([p for p in person_dir.iterdir()
                         if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
        if len(images) < args.min_images:
            print(f"[WARN] {norm_name}: need >= {args.min_images} images, found {len(images)}")
            continue

        feats = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] cannot read image: {img_path}")
                continue
            faces = app.get(img)
            if not faces:
                print(f"[WARN] no face in {img_path}")
                continue
            f = max(faces, key=lambda x: x.det_score)
            crop = norm_crop(img, f.kps, image_size=112)
            feat = app.models['recognition'].get_feat(crop)
            feat = feat / (np.linalg.norm(feat) + 1e-9)
            if feat.size == 512:
                feats.append(feat.astype(np.float32))
            else:
                print(f"[WARN] {img_path} embedding dim !=512 (got {feat.size})")

        if not feats:
            print(f"[WARN] {norm_name}: no valid faces")
            continue

        mean_feat = np.mean(np.stack(feats, axis=0), axis=0)
        mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-9)

        info = meta_map.get(norm_name, {"class": "", "mssv": ""})
        facebank.add(mean_feat.astype(np.float32), name=norm_name,
                     cls=info.get('class',''), mssv=info.get('mssv',''))
        print(f"[OK] enrolled {norm_name} with {len(feats)} images")

    facebank.save()
    # In bảng đối chiếu chỉ số → tên để kiểm tra nhanh
    print("\n[SUMMARY] Facebank entries:")
    for i, info in enumerate(facebank.meta):
        print(f"  {i:02d}  {info.get('name','')} | {info.get('class','')} | {info.get('mssv','')}")
    print(f"\nFacebank saved to: {facebank.dir}")

if __name__ == '__main__':
    main()

import argparse
from pathlib import Path
import csv
import json
import cv2
import numpy as np
import unicodedata as ud
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

def N(x: str) -> str:
    return ud.normalize("NFC", x or "").strip()

def _default_providers():
    avail = set(ort.get_available_providers())
    order = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    return [p for p in order if p in avail]

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

def try_detect_then_align(app: FaceAnalysis, img: np.ndarray, upscale: float|None=None) -> np.ndarray|None:
    """Trả về ảnh 112x112 đã align hoặc None nếu không thể."""
    src = img
    if upscale and upscale > 1.0:
        h, w = img.shape[:2]
        src = cv2.resize(img, (int(w*upscale), int(h*upscale)), interpolation=cv2.INTER_CUBIC)
    faces = app.get(src)
    if not faces:
        return None
    f = max(faces, key=lambda x: getattr(x, 'det_score', 0.0))
    crop = norm_crop(src, f.kps, image_size=112)
    return crop

def naive_to_112(img: np.ndarray) -> np.ndarray:
    """Fallback cuối: resize (center-pad/crop) về 112x112 nếu detector bó tay."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    # Fit shortest side to 112, keep aspect
    scale = 112.0 / min(h, w)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    # Center crop 112x112
    y0 = max(0, (nh - 112)//2); x0 = max(0, (nw - 112)//2)
    crop = resized[y0:y0+112, x0:x0+112]
    if crop.shape[0] != 112 or crop.shape[1] != 112:
        # pad if needed
        out = np.zeros((112,112,3), dtype=crop.dtype)
        out[:crop.shape[0], :crop.shape[1]] = crop
        crop = out
    return crop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--people_dir', type=str, required=True, help='data/people')
    ap.add_argument('--meta_csv', type=str, default='')
    ap.add_argument('--min_images', type=int, default=3)
    args = ap.parse_args()

    people_dir = Path(args.people_dir)
    meta_csv = Path(args.meta_csv) if args.meta_csv else None

    providers = _default_providers()
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640,640))

    # Load facebank
    from src.matching import FaceBank
    facebank = FaceBank()
    facebank.embeddings = np.zeros((0, 512), dtype=np.float32)
    facebank.meta = []
    facebank.save()  # reset sạch

    meta_map = read_meta_csv(meta_csv) if meta_csv else {}

    # Duyệt thư mục người (sort tên đã normalize)
    person_dirs = []
    for p in people_dir.iterdir():
        if p.is_dir():
            person_dirs.append((N(p.name), p))
    person_dirs.sort(key=lambda x: x[0])

    for norm_name, person_dir in person_dirs:
        # Ưu tiên ảnh đã align nếu có (tên file bắt đầu bằng 'align_')
        aligned_imgs = sorted([p for p in person_dir.iterdir()
                               if p.suffix.lower() in {'.jpg','.jpeg','.png'} and p.name.startswith('align_')])
        raw_imgs = sorted([p for p in person_dir.iterdir()
                           if p.suffix.lower() in {'.jpg','.jpeg','.png'} and not p.name.startswith('align_')])

        feats = []

        def embed_from_img(img_bgr: np.ndarray) -> np.ndarray|None:
            feat = app.models['recognition'].get_feat(img_bgr)
            if feat is None or feat.size != 512:
                return None
            feat = feat / (np.linalg.norm(feat) + 1e-9)
            return feat.astype(np.float32)

        # 1) Dùng ảnh align sẵn nếu có
        if aligned_imgs:
            for pth in aligned_imgs:
                img = cv2.imread(str(pth))
                if img is None: continue
                # Đảm bảo size 112x112
                if img.shape[:2] != (112,112):
                    img = cv2.resize(img, (112,112))
                f = embed_from_img(img)
                if f is not None: feats.append(f)

        # 2) Nếu chưa đủ, thử from raw: detect → align; nếu fail, upscale → detect; nếu vẫn fail, naive 112
        for pth in raw_imgs:
            if len(feats) >= args.min_images:
                break
            img = cv2.imread(str(pth))
            if img is None: 
                print(f"[WARN] cannot read image: {pth}")
                continue

            crop = try_detect_then_align(app, img, upscale=None)
            if crop is None:
                crop = try_detect_then_align(app, img, upscale=1.6)
            if crop is None:
                crop = try_detect_then_align(app, img, upscale=2.0)
            if crop is None:
                crop = naive_to_112(img)

            if crop is None:
                print(f"[WARN] {norm_name}: cannot prepare face from {pth.name}")
                continue

            f = embed_from_img(crop)
            if f is not None:
                feats.append(f)
            else:
                print(f"[WARN] {norm_name}: embedding failed on {pth.name}")

        if len(feats) < args.min_images:
            print(f"[WARN] {norm_name}: need >= {args.min_images} good images, got {len(feats)}")
            continue

        mean_feat = np.mean(np.stack(feats, axis=0), axis=0)
        mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-9)

        info = meta_map.get(norm_name, {"class": "", "mssv": ""})
        facebank.add(mean_feat.astype(np.float32), name=norm_name,
                     cls=info.get('class',''), mssv=info.get('mssv',''))
        print(f"[OK] enrolled {norm_name} with {len(feats)} images")

    facebank.save()
    print("\n[SUMMARY] Facebank entries:")
    for i, info in enumerate(facebank.meta):
        print(f"  {i:02d}  {info.get('name','')} | {info.get('class','')} | {info.get('mssv','')}")
    print(f"\nFacebank saved to: {facebank.dir}")

if __name__ == '__main__':
    main()



```markdown
# Face Attendance (Realtime) — SCRFD → Alignment → ArcFace → Matching


**Goal:** Realtime attendance on webcam/USB/IP camera for ~30 people (demo with 5). If a face is recognized, show **name, class, MSSV**; else show **Unknown**.


## 1) Pipeline


Camera Frame → (1) Detection (SCRFD) → (2) Alignment (Umeyama / ArcFace template) → (3) Embedding (ArcFace 512‑D) → (4) Matching & Display


- **Detection**: SCRFD (InsightFace) → bbox + 5 landmarks
- **Alignment**: `norm_crop` (Umeyama) to 112×112 RGB
- **Embedding**: ArcFace model (L2‑normalized 512‑D)
- **Matching**: cosine similarity vs **facebank** (your enrolled students)


## 2) Install


> Python 3.9–3.11 recommended. Create a venv first.


```bash
pip install -r requirements.txt
```


### Apple Silicon (M‑series)
On macOS ARM, `onnxruntime` wheel differs. If you hit errors:
```bash
pip uninstall -y onnxruntime
pip install onnxruntime-silicon
```


## 3) Quick start (no dataset yet)
Run the realtime app; it will detect faces and label them as **Unknown** until you register students.
```bash
python -m src.app --source 0
```
- `--source 0` = default webcam; also accepts a file path or RTSP/HTTP URL.
- Press **Q** to quit.


## 4) Register faces (later)
Prepare images under `data/people/<name>/img_*.jpg`. Optional CSV for metadata:
```
name,class,mssv
Alice,DS-A,123456
Bob,DS-B,234567
```
Then run:
```bash
python -m scripts.register_faces --people_dir data/people --meta_csv data/people/meta.csv
```
This builds `data/facebank/{embeddings.npy, meta.json}`.


## 5) Config
Edit thresholds and model bundles in `src/config.py`.


## 6) Notes
- First run downloads InsightFace models (~200MB). They are cached in `~/.insightface`.
- All modules are separated to match the exact 4‑stage pipeline.
- Works CPU‑only. If you have GPU/CUDA, InsightFace will use it automatically.


---
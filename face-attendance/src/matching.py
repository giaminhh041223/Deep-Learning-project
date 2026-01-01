from __future__ import annotations
import json
import threading
from pathlib import Path
import numpy as np
from .config import CFG

class FaceBank:
    def __init__(self, facebank_dir: Path | None = None):
        self.dir = (facebank_dir or CFG.facebank_dir)
        self.emb_path = self.dir / "embeddings.npy"
        self.meta_path = self.dir / "meta.json"
        self.embeddings = None  
        self.meta = []          
        self.lock = threading.Lock()
        self.load()

    def _empty_emb(self):
        return np.zeros((0, 512), dtype=np.float32)

    def _sanitize_emb_matrix(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            if arr.size == 512:
                return arr.reshape(1, 512).astype(np.float32)
            return self._empty_emb()

        if arr.ndim == 2:
            h, w = arr.shape
            if w == 512:
                return arr.astype(np.float32)
            if h == 512 and w == 1:
                return arr.reshape(1, 512).astype(np.float32)
            if h == 512 and w > 1:
                arr_t = arr.T
                if arr_t.shape[1] == 512:
                    return arr_t.astype(np.float32)
            if w == 1 and h > 1:
                if h % 512 == 0:
                    n = h // 512
                    return arr.reshape(n, 512).astype(np.float32)
        return self._empty_emb()

    def load(self):
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            if self.emb_path.exists() and self.emb_path.stat().st_size > 0:
                try:
                    raw = np.load(self.emb_path, allow_pickle=False)
                    self.embeddings = self._sanitize_emb_matrix(raw)
                except Exception:
                    self.embeddings = self._empty_emb()
            else:
                self.embeddings = self._empty_emb()
            if self.meta_path.exists() and self.meta_path.stat().st_size > 0:
                try:
                    self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                    if not isinstance(self.meta, list):
                        self.meta = []
                except Exception:
                    self.meta = []
            else:
                self.meta = []
            if self.embeddings.shape[0] < len(self.meta):
                self.meta = self.meta[: self.embeddings.shape[0]]

    def save(self):
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            np.save(self.emb_path, self.embeddings if self.embeddings is not None else self._empty_emb())
            self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[FaceBank] Saved {len(self.meta)} faces to disk.")

    def add(self, emb: np.ndarray, name: str, cls: str = "", mssv: str = ""):
        emb = np.asarray(emb).astype(np.float32).reshape(-1)
        if emb.size != 512:
            return  
        emb = (emb / (np.linalg.norm(emb) + 1e-9)).astype(np.float32)
        with self.lock:
            if self.embeddings is None or self.embeddings.size == 0:
                self.embeddings = emb.reshape(1, 512)
            else:
                self.embeddings = np.vstack([self.embeddings, emb.reshape(1, 512)])
            self.meta.append({"name": name, "class": cls, "mssv": mssv})
        self.save()

    def match(self, emb: np.ndarray, threshold: float) -> tuple[str, float, dict]:
        if self.embeddings is None or self.embeddings.size == 0:
            return "Unknown", 0.0, {}
        
        emb = np.asarray(emb).astype(np.float32).reshape(-1)
        if emb.size != 512:
            return "Unknown", 0.0, {}
        sims = self.embeddings @ emb  # (N,)
        idx = int(np.argmax(sims))
        smax = float(sims[idx])
        
        if smax >= threshold and idx < len(self.meta):
            info = self.meta[idx]
            parts = [info.get("name", "").strip(), info.get("class", "").strip(), info.get("mssv", "").strip()]
            label = " | ".join([p for p in parts if p]) or "Unknown"
            return label, smax, info
            
        return "Unknown", smax, {}

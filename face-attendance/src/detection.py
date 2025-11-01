import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from .config import CFG

def _default_providers():
    avail = set(ort.get_available_providers())
    order = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    return [p for p in order if p in avail]

class Detector:
    def __init__(self, det_size: int | None = None, providers=None):
        self.det_size = det_size or CFG.det_size
        providers = providers or _default_providers()
        self.app = FaceAnalysis(name=CFG.bundle, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))

    def detect(self, img_bgr: np.ndarray, min_face: int | None = None):
        faces = self.app.get(img_bgr)
        out = []
        min_face = min_face or CFG.min_face
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            if max(0, x2 - x1) < min_face or max(0, y2 - y1) < min_face:
                continue
            out.append({
                "bbox": f.bbox.astype(float),
                "kps": f.kps.astype(float),  # (5,2)
                "det_score": float(getattr(f, 'det_score', 1.0)),
            })
        return out
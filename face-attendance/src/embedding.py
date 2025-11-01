import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from .config import CFG

def _default_providers():
    avail = set(ort.get_available_providers())
    order = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    return [p for p in order if p in avail]

class Embedder:
    """
    Dùng cùng bundle FaceAnalysis (buffalo_l) để lấy model recognition
    và truyền providers rõ ràng cho ONNX Runtime trên macOS ARM.
    """
    def __init__(self, providers=None):
        providers = providers or _default_providers()
        self.app = FaceAnalysis(name=CFG.bundle, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(CFG.det_size, CFG.det_size))
        self.rec = self.app.models['recognition']
        if self.rec is None:
            raise RuntimeError("Recognition model not loaded from FaceAnalysis bundle.")

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        feat = self.rec.get_feat(aligned_bgr_112)
        feat = feat / (np.linalg.norm(feat) + 1e-9)
        return feat.astype(np.float32)

import cv2
import numpy as np
from .template_112 import TEMPLATE_112

class FaceAligner:
    """
    FaceAligner: Aligns faces using 5-point landmarks and a fixed 112x112 template.
    Input:
        - img_bgr: Original face image (BGR)
        - landmarks: np.ndarray of shape (5,2)
    Output:
        - aligned: 112x112 aligned face image
    """

    def __init__(self, template=TEMPLATE_112, size=(112, 112)):
        self.template = template.astype(np.float32)
        self.size = size

    def align(self, img_bgr, landmarks):
        """Align the face image based on given 5-point landmarks."""
        if landmarks is None or len(landmarks) != 5:
            raise ValueError("Expected 5 landmark points for alignment.")
        
        lmk = np.array(landmarks, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(lmk, self.template, method=cv2.LMEDS)
        
        if M is None:
            print("[WARN] Affine matrix estimation failed.")
            return None
        
        aligned = cv2.warpAffine(img_bgr, M, self.size, flags=cv2.INTER_LINEAR)
        return aligned

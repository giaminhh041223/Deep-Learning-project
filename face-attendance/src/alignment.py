import numpy as np
from insightface.utils.face_align import norm_crop


class Aligner:
    def __init__(self, image_size: int = 112):
        self.image_size = image_size
        
    def align(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        crop = norm_crop(img_bgr, landmark=kps, image_size=self.image_size)
        return crop

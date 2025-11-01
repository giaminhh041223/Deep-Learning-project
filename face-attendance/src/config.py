from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Model bundle from InsightFace (detector: SCRFD, recognizer: ArcFace)
    bundle: str = "buffalo_l" # includes SCRFD + ArcFace

    # Matching
    threshold: float = 0.35 # cosine similarity threshold (tune 0.3~0.5)
    facebank_dir: Path = Path("data/facebank")

    # Detection
    det_size: int = 640
    min_face: int = 80 # ignore faces smaller than this (pixels)

    # Drawing / UI
    draw_fps: bool = True

CFG = Config()
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    bundle: str = "buffalo_l" 
    threshold: float = 0.75 
    facebank_dir: Path = Path("data/facebank")

    # Detection
    det_size: int = 640
    min_face: int = 80
    draw_fps: bool = True

CFG = Config()

# Face Alignment Module

This module performs face alignment using 5-point facial landmarks.
It takes as input the original face image and the corresponding 5 landmark coordinates,
and outputs a normalized 112x112 face crop aligned to a standard template.

## Input
- `img_bgr`: Original image (BGR)
- `landmarks`: np.array of shape (5,2), [L-eye, R-eye, Nose, L-mouth, R-mouth]

## Output
- 112x112 aligned face (np.ndarray)

## Run demo
```bash
python src/demo_image.py --img examples/input/1.jpg

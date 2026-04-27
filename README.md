# NudeNet — Lightweight Nudity Detection

YOLOv8-based nudity detector & censor — runs fully on CPU via ONNX Runtime.

## Install

```bash
pip install nudenet
```

## Quick start

```python
from nudenet import NudeDetector

detector = NudeDetector()           # uses bundled 320n.onnx by default

# --- Single image ---
detections = detector.detect("photo.jpg")
# [{'class': 'BELLY_EXPOSED', 'score': 0.82, 'box': [64, 182, 49, 51]}, ...]

# --- Batch ---
results = detector.detect_batch(["a.jpg", "b.jpg"])

# --- Censor (black box, blur, or pixelate) ---
detector.censor("photo.jpg")                          # black box (default)
detector.censor("photo.jpg", censor_style="blur")     # Gaussian blur
detector.censor("photo.jpg", censor_style="pixel")    # pixelate / mosaic

# --- Censor specific classes only ---
detector.censor("photo.jpg", classes=["FACE_FEMALE", "FACE_MALE"])

# --- Tune detection thresholds ---
detector.detect("photo.jpg", score_threshold=0.4, nms_threshold=0.5)
```

## Accepted input types

All methods accept any of the following:

| Type | Example |
|---|---|
| `str` file path | `"photo.jpg"` |
| `np.ndarray` (BGR) | `cv2.imread("photo.jpg")` |
| `bytes` | `open("photo.jpg","rb").read()` |
| `BufferedReader` | `open("photo.jpg","rb")` |

## Models

| Model | Resolution | Based on | ONNX |
|---|---|---|---|
| **320n** *(bundled)* | 320×320 | YOLOv8n | included |
| 640m *(download)* | 640×640 | YOLOv8m | [link](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx) |

```python
# Use the larger 640m model
detector = NudeDetector(model_path="640m.onnx", inference_resolution=640)
```

## All detectable labels

```python
from nudenet import LABELS
print(LABELS)
# ['FEMALE_GENITALIA_COVERED', 'FACE_FEMALE', 'BUTTOCKS_EXPOSED', ...]
```

## License

MIT

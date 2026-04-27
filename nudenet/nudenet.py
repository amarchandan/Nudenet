"""
nudenet.nudenet
~~~~~~~~~~~~~~~

Core detection and censoring logic built on top of a YOLOv8-based ONNX model.

Fixes over original v3.4.2
---------------------------
* _read_image: channel-aware conversion (grayscale / RGBA / BGR) instead of
  blindly assuming RGBA for every image.
* censor(): accepts all input types (str, ndarray, bytes, BufferedReader) via
  the shared _load_mat() helper — was broken for non-string inputs.
* detect_batch(): per-image output slicing now correctly handles both batched
  (N, 84, 8400) and un-batched (84, 8400) ONNX output shapes.
* score_threshold / nms_threshold are now user-configurable on every public
  method instead of being hardcoded.
* censor() supports three censor styles: 'black', 'blur', 'pixel'.
* Meaningful exceptions and logging throughout instead of silent crashes.
"""

import _io
import logging
import os
from typing import List, Optional, Union

import cv2
import numpy as np
import onnxruntime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class labels (fixed — same order as model output)
# ---------------------------------------------------------------------------
LABELS: List[str] = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

ImageInput = Union[str, np.ndarray, bytes, _io.BufferedReader]
_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "320n.onnx")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_mat(image_input: ImageInput) -> np.ndarray:
    """Return a BGR uint8 ndarray from any supported image input."""
    if isinstance(image_input, np.ndarray):
        return image_input.copy()

    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        mat = cv2.imread(image_input)
        if mat is None:
            raise ValueError(f"cv2 could not read: {image_input}")
        return mat

    if isinstance(image_input, bytes):
        buf = np.frombuffer(image_input, np.uint8)
    elif isinstance(image_input, _io.BufferedReader):
        buf = np.frombuffer(image_input.read(), np.uint8)
    else:
        raise TypeError(
            f"Unsupported type {type(image_input).__name__}. "
            "Expected str, np.ndarray, bytes, or BufferedReader."
        )

    mat = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if mat is None:
        raise ValueError("Failed to decode image from buffer.")
    return mat


def _read_image(image_input: ImageInput, target_size: int = 320):
    """
    Load and preprocess one image for ONNX inference.

    Returns
    -------
    (input_blob, x_ratio, y_ratio, x_pad, y_pad, orig_w, orig_h)
    """
    mat = _load_mat(image_input)

    orig_h, orig_w = mat.shape[:2]

    # FIX: channel-aware colour conversion
    if mat.ndim == 2:                          # grayscale
        mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
    elif mat.shape[2] == 4:                    # RGBA / BGRA
        mat = cv2.cvtColor(mat, cv2.COLOR_BGRA2BGR)
    # else already BGR — nothing to do

    max_side = max(mat.shape[:2])
    x_pad = max_side - mat.shape[1]
    y_pad = max_side - mat.shape[0]
    x_ratio = max_side / mat.shape[1]
    y_ratio = max_side / mat.shape[0]

    mat_pad = cv2.copyMakeBorder(mat, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)

    blob = cv2.dnn.blobFromImage(
        mat_pad,
        1 / 255.0,
        (target_size, target_size),
        (0, 0, 0),
        swapRB=True,
        crop=False,
    )
    return blob, x_ratio, y_ratio, x_pad, y_pad, orig_w, orig_h


def _postprocess(
    raw_output,
    x_pad: int,
    y_pad: int,
    x_ratio: float,
    y_ratio: float,
    orig_w: int,
    orig_h: int,
    model_w: int,
    model_h: int,
    score_threshold: float,
    nms_threshold: float,
) -> List[dict]:
    """
    Convert raw ONNX output into a list of detection dicts.

    Handles both batched  (1, 84, 8400) and un-batched (84, 8400) shapes.
    """
    arr = raw_output[0]
    if arr.ndim == 3:        # (batch, channels, anchors) → drop batch dim
        arr = arr[0]
    outputs = np.transpose(arr)   # (anchors, channels)

    boxes, scores, class_ids = [], [], []

    for row in outputs:
        class_scores = row[4:]
        max_score = float(class_scores.max())
        if max_score < score_threshold:
            continue

        class_id = int(class_scores.argmax())
        cx, cy, w, h = row[:4]

        # centre → top-left, then scale to padded space
        x = (cx - w / 2) * (orig_w + x_pad) / model_w
        y = (cy - h / 2) * (orig_h + y_pad) / model_h
        w = w * (orig_w + x_pad) / model_w
        h = h * (orig_h + y_pad) / model_h

        # clip to image
        x = float(np.clip(x, 0, orig_w))
        y = float(np.clip(y, 0, orig_h))
        w = float(np.clip(w, 0, orig_w - x))
        h = float(np.clip(h, 0, orig_h - y))

        boxes.append([x, y, w, h])
        scores.append(max_score)
        class_ids.append(class_id)

    if not boxes:
        return []

    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
    return [
        {
            "class": LABELS[class_ids[i]],
            "score": round(float(scores[i]), 6),
            "box": [int(v) for v in boxes[i]],
        }
        for i in keep
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NudeDetector:
    """
    Lightweight nudity detector powered by a YOLOv8 ONNX model.

    Parameters
    ----------
    model_path : str, optional
        Path to a custom ``.onnx`` model file.
        Defaults to the bundled ``320n.onnx``.
    inference_resolution : int
        Square input size fed to the model (default 320).
        Use 640 together with the 640m model for higher accuracy.

    Examples
    --------
    >>> from nudenet import NudeDetector
    >>> detector = NudeDetector()
    >>> detector.detect("photo.jpg")
    [{'class': 'BELLY_EXPOSED', 'score': 0.82, 'box': [64, 182, 49, 51]}, ...]
    >>> detector.censor("photo.jpg", censor_style="blur")
    'photo_censored.jpg'
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        inference_resolution: int = 320,
    ) -> None:
        resolved = model_path or _DEFAULT_MODEL
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"ONNX model not found: {resolved}")

        logger.debug("Loading model %s at %dpx", resolved, inference_resolution)
        self._session = onnxruntime.InferenceSession(
            resolved,
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self.input_width: int = inference_resolution
        self.input_height: int = inference_resolution

    # ------------------------------------------------------------------
    # Single-image detection
    # ------------------------------------------------------------------

    def detect(
        self,
        image: ImageInput,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> List[dict]:
        """
        Detect nudity-related regions in a single image.

        Parameters
        ----------
        image :
            File path, ``np.ndarray`` (BGR), ``bytes``, or ``BufferedReader``.
        score_threshold :
            Minimum confidence to keep a detection (default 0.25).
        nms_threshold :
            IoU overlap threshold for Non-Maximum Suppression (default 0.45).

        Returns
        -------
        list of dict
            ``[{'class': str, 'score': float, 'box': [x, y, w, h]}, ...]``
        """
        blob, x_ratio, y_ratio, x_pad, y_pad, orig_w, orig_h = _read_image(
            image, self.input_width
        )
        outputs = self._session.run(None, {self._input_name: blob})
        return _postprocess(
            outputs, x_pad, y_pad, x_ratio, y_ratio,
            orig_w, orig_h, self.input_width, self.input_height,
            score_threshold, nms_threshold,
        )

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        images: List[ImageInput],
        batch_size: int = 4,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> List[List[dict]]:
        """
        Detect nudity-related regions in multiple images efficiently.

        Parameters
        ----------
        images :
            List of images (any supported input type).
        batch_size :
            Number of images processed per ONNX forward pass (default 4).
        score_threshold :
            Minimum confidence to keep a detection.
        nms_threshold :
            IoU threshold for Non-Maximum Suppression.

        Returns
        -------
        list of list of dict
            One detection list per input image, in the same order.
        """
        all_detections: List[List[dict]] = []

        for start in range(0, len(images), batch_size):
            chunk = images[start: start + batch_size]
            blobs, metas = [], []

            for img in chunk:
                blob, *meta = _read_image(img, self.input_width)
                blobs.append(blob)
                metas.append(meta)  # (x_ratio, y_ratio, x_pad, y_pad, orig_w, orig_h)

            batch_blob = np.vstack(blobs)
            outputs = self._session.run(None, {self._input_name: batch_blob})

            for j, (x_ratio, y_ratio, x_pad, y_pad, orig_w, orig_h) in enumerate(metas):
                # FIX: correct per-image slice from batched output
                single = [outputs[0][j: j + 1]]
                detections = _postprocess(
                    single, x_pad, y_pad, x_ratio, y_ratio,
                    orig_w, orig_h, self.input_width, self.input_height,
                    score_threshold, nms_threshold,
                )
                all_detections.append(detections)

        return all_detections

    # ------------------------------------------------------------------
    # Censoring
    # ------------------------------------------------------------------

    def censor(
        self,
        image: ImageInput,
        classes: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        censor_style: str = "black",
        blur_factor: int = 15,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> str:
        """
        Detect and censor nudity-related regions, then save the result.

        Parameters
        ----------
        image :
            Input image — file path, ``np.ndarray``, ``bytes``, or
            ``BufferedReader``.
        classes :
            Restrict censoring to these label strings.  ``None`` / ``[]``
            censors everything that is detected.
        output_path :
            Destination file path.  Auto-generated when omitted.
        censor_style :
            ``'black'``  — solid black rectangle (default).
            ``'blur'``   — Gaussian blur.
            ``'pixel'``  — pixelate (mosaic).
        blur_factor :
            Kernel / block size used by ``'blur'`` and ``'pixel'`` styles.
            Must be a positive integer (default 15).
        score_threshold :
            Minimum confidence for a detection to be censored.
        nms_threshold :
            IoU threshold for Non-Maximum Suppression.

        Returns
        -------
        str
            Absolute path to the saved censored image.

        Raises
        ------
        ValueError
            If ``censor_style`` is unrecognised or the image cannot be saved.
        """
        if censor_style not in ("black", "blur", "pixel"):
            raise ValueError(
                f"Unknown censor_style '{censor_style}'. "
                "Choose from: 'black', 'blur', 'pixel'."
            )

        detections = self.detect(image, score_threshold, nms_threshold)
        if classes:
            detections = [d for d in detections if d["class"] in classes]

        # FIX: _load_mat() works for all input types, not just str paths
        mat = _load_mat(image)

        for det in detections:
            x, y, w, h = det["box"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(mat.shape[1], x + w), min(mat.shape[0], y + h)
            roi = mat[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if censor_style == "black":
                mat[y1:y2, x1:x2] = 0

            elif censor_style == "blur":
                k = max(1, blur_factor | 1)        # ensure odd
                mat[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

            else:  # pixel
                factor = max(1, blur_factor)
                sh = max(1, (y2 - y1) // factor)
                sw = max(1, (x2 - x1) // factor)
                small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
                mat[y1:y2, x1:x2] = cv2.resize(
                    small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
                )

        # Determine output path
        if not output_path:
            if isinstance(image, str):
                base, ext = os.path.splitext(image)
                output_path = f"{base}_censored{ext}"
            else:
                output_path = "censored_output.jpg"

        if not cv2.imwrite(output_path, mat):
            raise ValueError(f"Failed to write censored image to: {output_path}")

        logger.debug("Censored image saved → %s", output_path)
        return output_path

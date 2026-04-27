import os
import _io
import logging
import cv2
import numpy as np
import onnxruntime
from typing import List, Union, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
__labels = [
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


# ---------------------------------------------------------------------------
# FIX 1: _read_image — correct channel handling (was hardcoded RGBA2BGR)
# ---------------------------------------------------------------------------
def _read_image(
    image_path: Union[str, np.ndarray, bytes, _io.BufferedReader],
    target_size: int = 320,
):
    """
    Load an image from various input types and preprocess it for model inference.

    Args:
        image_path: File path (str), numpy array, raw bytes, or BufferedReader.
        target_size: Model input resolution (default 320).

    Returns:
        Tuple of (input_blob, x_ratio, y_ratio, x_pad, y_pad,
                  image_original_width, image_original_height)

    Raises:
        ValueError: If image_path type is unsupported or image cannot be loaded.
    """
    # --- Load raw mat ---
    if isinstance(image_path, str):
        if not os.path.isfile(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        mat = cv2.imread(image_path)
        if mat is None:
            raise ValueError(f"cv2.imread failed to load image: {image_path}")
    elif isinstance(image_path, np.ndarray):
        mat = image_path.copy()
    elif isinstance(image_path, bytes):
        arr = np.frombuffer(image_path, np.uint8)
        mat = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if mat is None:
            raise ValueError("Failed to decode image from bytes.")
    elif isinstance(image_path, _io.BufferedReader):
        arr = np.frombuffer(image_path.read(), np.uint8)
        mat = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if mat is None:
            raise ValueError("Failed to decode image from BufferedReader.")
    else:
        raise ValueError(
            "image_path must be a str path, np.ndarray, bytes, or BufferedReader. "
            f"Got: {type(image_path)}"
        )

    if mat is None or mat.size == 0:
        raise ValueError("Loaded image is empty or invalid.")

    image_original_height, image_original_width = mat.shape[:2]

    # FIX 1: Handle channels correctly instead of blindly assuming RGBA
    if mat.ndim == 2:
        # Grayscale → BGR
        mat_c3 = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
    elif mat.shape[2] == 4:
        # RGBA → BGR
        mat_c3 = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)
    else:
        # Already BGR (3-channel) — most JPEG/PNG images
        mat_c3 = mat

    # --- Letterbox padding to make image square ---
    max_size = max(mat_c3.shape[:2])
    x_pad = max_size - mat_c3.shape[1]
    x_ratio = max_size / mat_c3.shape[1]
    y_pad = max_size - mat_c3.shape[0]
    y_ratio = max_size / mat_c3.shape[0]

    mat_pad = cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)

    input_blob = cv2.dnn.blobFromImage(
        mat_pad,
        1 / 255.0,
        (target_size, target_size),
        (0, 0, 0),
        swapRB=True,
        crop=False,
    )

    return (
        input_blob,
        x_ratio,
        y_ratio,
        x_pad,
        y_pad,
        image_original_width,
        image_original_height,
    )


# ---------------------------------------------------------------------------
# FIX 2: _postprocess — user-configurable NMS thresholds
# ---------------------------------------------------------------------------
def _postprocess(
    output,
    x_pad: int,
    y_pad: int,
    x_ratio: float,
    y_ratio: float,
    image_original_width: int,
    image_original_height: int,
    model_width: int,
    model_height: int,
    score_threshold: float = 0.25,
    nms_threshold: float = 0.45,
) -> List[dict]:
    """
    Convert raw ONNX model output into bounding box detections.

    Args:
        output: Raw ONNX model output list.
        score_threshold: Minimum confidence to keep a detection.
        nms_threshold: IoU threshold for Non-Maximum Suppression.

    Returns:
        List of dicts with keys 'class', 'score', 'box'.
    """
    # FIX 3: Safe squeeze — handles both batched (N,84,8400) and single (84,8400) outputs
    raw = output[0]
    if raw.ndim == 3:
        raw = raw[0]  # take first item if batch dim present
    outputs = np.transpose(raw)

    rows = outputs.shape[0]
    boxes, scores, class_ids = [], [], []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = float(np.amax(classes_scores))

        if max_score >= score_threshold:
            class_id = int(np.argmax(classes_scores))
            x, y, w, h = outputs[i][0:4]

            # Center → top-left
            x = x - w / 2
            y = y - h / 2

            # Scale to padded image space
            x = x * (image_original_width + x_pad) / model_width
            y = y * (image_original_height + y_pad) / model_height
            w = w * (image_original_width + x_pad) / model_width
            h = h * (image_original_height + y_pad) / model_height

            # Clip to image boundaries
            x = max(0.0, min(x, float(image_original_width)))
            y = max(0.0, min(y, float(image_original_height)))
            w = min(w, float(image_original_width) - x)
            h = min(h, float(image_original_height) - y)

            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, w, h])

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)

    detections = []
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        detections.append(
            {
                "class": __labels[class_ids[i]],
                "score": float(scores[i]),
                "box": [int(x), int(y), int(w), int(h)],
            }
        )

    return detections


# ---------------------------------------------------------------------------
# Helper: load image as numpy array from any supported input type
# ---------------------------------------------------------------------------
def _load_mat(image_input: Union[str, np.ndarray, bytes, _io.BufferedReader]) -> np.ndarray:
    """
    Load image as a BGR numpy array from any supported input type.
    Used by censor() to support all input types, not just file paths.
    """
    if isinstance(image_input, np.ndarray):
        return image_input.copy()
    elif isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise ValueError(f"Image file not found: {image_input}")
        mat = cv2.imread(image_input)
        if mat is None:
            raise ValueError(f"cv2.imread failed: {image_input}")
        return mat
    elif isinstance(image_input, bytes):
        arr = np.frombuffer(image_input, np.uint8)
        mat = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if mat is None:
            raise ValueError("Failed to decode image from bytes.")
        return mat
    elif isinstance(image_input, _io.BufferedReader):
        arr = np.frombuffer(image_input.read(), np.uint8)
        mat = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if mat is None:
            raise ValueError("Failed to decode image from BufferedReader.")
        return mat
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class NudeDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        inference_resolution: int = 320,
    ):
        """
        Initialize the NudeDetector.

        Args:
            model_path: Path to a custom ONNX model. Defaults to bundled 320n.onnx.
            inference_resolution: Input resolution to use for inference (e.g. 320 or 640).
        """
        resolved_model = (
            model_path
            if model_path
            else os.path.join(os.path.dirname(__file__), "320n.onnx")
        )

        if not os.path.isfile(resolved_model):
            raise FileNotFoundError(f"ONNX model not found at: {resolved_model}")

        logger.info(f"Loading model: {resolved_model} at resolution {inference_resolution}")

        # CPU-only inference (no GPU)
        self.onnx_session = onnxruntime.InferenceSession(
            resolved_model,
            providers=["CPUExecutionProvider"],
        )

        model_inputs = self.onnx_session.get_inputs()
        self.input_width = inference_resolution
        self.input_height = inference_resolution
        self.input_name = model_inputs[0].name

    def detect(
        self,
        image_path: Union[str, np.ndarray, bytes, _io.BufferedReader],
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> List[dict]:
        """
        Detect nudity-related classes in a single image.

        Args:
            image_path: Image as file path, numpy array, bytes, or BufferedReader.
            score_threshold: Minimum confidence score for a detection to be kept.
            nms_threshold: IoU threshold for Non-Maximum Suppression.

        Returns:
            List of dicts: [{'class': str, 'score': float, 'box': [x, y, w, h]}, ...]
        """
        (
            preprocessed_image,
            x_ratio,
            y_ratio,
            x_pad,
            y_pad,
            image_original_width,
            image_original_height,
        ) = _read_image(image_path, self.input_width)

        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})

        return _postprocess(
            outputs,
            x_pad,
            y_pad,
            x_ratio,
            y_ratio,
            image_original_width,
            image_original_height,
            self.input_width,
            self.input_height,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )

    def detect_batch(
        self,
        image_paths: List[Union[str, np.ndarray, bytes, _io.BufferedReader]],
        batch_size: int = 4,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> List[List[dict]]:
        """
        Perform batch detection on a list of images.

        Args:
            image_paths: List of images (file paths, arrays, bytes, or BufferedReaders).
            batch_size: Number of images per inference batch.
            score_threshold: Minimum confidence score for detections.
            nms_threshold: IoU threshold for Non-Maximum Suppression.

        Returns:
            List of detection lists, one per input image.
        """
        all_detections = []

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i: i + batch_size]
            batch_inputs = []
            batch_metadata = []

            for image_path in batch:
                (
                    preprocessed_image,
                    x_ratio,
                    y_ratio,
                    x_pad,
                    y_pad,
                    image_original_width,
                    image_original_height,
                ) = _read_image(image_path, self.input_width)
                batch_inputs.append(preprocessed_image)
                batch_metadata.append(
                    (x_ratio, y_ratio, x_pad, y_pad, image_original_width, image_original_height)
                )

            batch_input = np.vstack(batch_inputs)
            outputs = self.onnx_session.run(None, {self.input_name: batch_input})

            # FIX 3: Correct per-image output slicing for batch
            # outputs[0] shape: (batch_size, num_classes+4, num_anchors)
            for j, metadata in enumerate(batch_metadata):
                (x_ratio, y_ratio, x_pad, y_pad, image_original_width, image_original_height) = metadata

                # Extract single image output and wrap in list to match _postprocess signature
                single_output = [outputs[0][j: j + 1]]

                detections = _postprocess(
                    single_output,
                    x_pad,
                    y_pad,
                    x_ratio,
                    y_ratio,
                    image_original_width,
                    image_original_height,
                    self.input_width,
                    self.input_height,
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                )
                all_detections.append(detections)

        return all_detections

    def censor(
        self,
        image_path: Union[str, np.ndarray, bytes, _io.BufferedReader],
        classes: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        censor_style: str = "black",
        blur_factor: int = 15,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> str:
        """
        Detect and censor nudity-related regions in an image.

        Args:
            image_path: Image as file path, numpy array, bytes, or BufferedReader.
            classes: List of class names to censor. Censors all detected classes if empty/None.
            output_path: Where to save the censored image. Auto-generated if not provided.
            censor_style: 'black' for black box, 'blur' for Gaussian blur, 'pixel' for pixelate.
            blur_factor: Blur kernel size (used for 'blur' and 'pixel' styles). Must be odd.
            score_threshold: Minimum confidence for detections.
            nms_threshold: IoU threshold for Non-Maximum Suppression.

        Returns:
            Path to the saved censored image.

        Raises:
            ValueError: If image cannot be loaded, output_path is invalid, or style is unknown.
        """
        detections = self.detect(image_path, score_threshold, nms_threshold)

        if classes:
            detections = [d for d in detections if d["class"] in classes]

        # FIX 4: censor() now supports all input types, not just string paths
        img = _load_mat(image_path)

        for detection in detections:
            x, y, w, h = detection["box"]
            # Ensure ROI is within image bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
            roi = img[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            if censor_style == "black":
                img[y1:y2, x1:x2] = 0

            elif censor_style == "blur":
                # Ensure kernel is odd and at least 1
                k = max(1, blur_factor | 1)
                img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

            elif censor_style == "pixel":
                # Pixelate by downscaling then upscaling
                small_h = max(1, (y2 - y1) // blur_factor)
                small_w = max(1, (x2 - x1) // blur_factor)
                small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

            else:
                raise ValueError(
                    f"Unknown censor_style '{censor_style}'. Choose from: 'black', 'blur', 'pixel'."
                )

        # Determine output path
        if not output_path:
            if isinstance(image_path, str):
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_censored{ext}"
            else:
                output_path = "censored_output.jpg"

        saved = cv2.imwrite(output_path, img)
        if not saved:
            raise ValueError(f"Failed to save censored image to: {output_path}")

        logger.info(f"Censored image saved to: {output_path}")
        return output_path
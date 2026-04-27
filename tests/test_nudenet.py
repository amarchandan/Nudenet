"""
Basic unit tests for NudeDetector.
Run with:  pytest tests/ -v
"""

import io
import os
import tempfile

import cv2
import numpy as np
import pytest

from nudenet import NudeDetector, LABELS


@pytest.fixture(scope="module")
def detector():
    return NudeDetector()


@pytest.fixture()
def blank_image_path(tmp_path):
    """Write a small blank JPEG and return its path."""
    p = tmp_path / "blank.jpg"
    cv2.imwrite(str(p), np.zeros((320, 320, 3), dtype=np.uint8))
    return str(p)


@pytest.fixture()
def blank_ndarray():
    return np.zeros((320, 320, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Smoke tests — just check it runs without crashing
# ---------------------------------------------------------------------------

class TestDetect:
    def test_str_path(self, detector, blank_image_path):
        result = detector.detect(blank_image_path)
        assert isinstance(result, list)

    def test_ndarray(self, detector, blank_ndarray):
        result = detector.detect(blank_ndarray)
        assert isinstance(result, list)

    def test_bytes(self, detector, blank_image_path):
        with open(blank_image_path, "rb") as f:
            data = f.read()
        result = detector.detect(data)
        assert isinstance(result, list)

    def test_buffered_reader(self, detector, blank_image_path):
        with open(blank_image_path, "rb") as f:
            result = detector.detect(f)
        assert isinstance(result, list)

    def test_rgba_image(self, detector, tmp_path):
        """RGBA image must not crash."""
        p = tmp_path / "rgba.png"
        rgba = np.zeros((320, 320, 4), dtype=np.uint8)
        cv2.imwrite(str(p), rgba)
        assert isinstance(detector.detect(str(p)), list)

    def test_grayscale_image(self, detector, tmp_path):
        """Grayscale image must not crash."""
        p = tmp_path / "gray.jpg"
        cv2.imwrite(str(p), np.zeros((320, 320), dtype=np.uint8))
        assert isinstance(detector.detect(str(p)), list)

    def test_invalid_path_raises(self, detector):
        with pytest.raises(FileNotFoundError):
            detector.detect("nonexistent_file.jpg")

    def test_invalid_type_raises(self, detector):
        with pytest.raises(TypeError):
            detector.detect(12345)

    def test_detection_keys(self, detector, blank_ndarray):
        # blank image → no detections, but if there were, they must have these keys
        for det in detector.detect(blank_ndarray):
            assert {"class", "score", "box"} <= det.keys()
            assert det["class"] in LABELS
            assert 0.0 <= det["score"] <= 1.0
            assert len(det["box"]) == 4


class TestDetectBatch:
    def test_returns_one_list_per_image(self, detector, blank_image_path):
        paths = [blank_image_path] * 3
        results = detector.detect_batch(paths)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, list)

    def test_batch_equals_single(self, detector, blank_ndarray):
        single = detector.detect(blank_ndarray)
        batch = detector.detect_batch([blank_ndarray])
        assert single == batch[0]

    def test_empty_list(self, detector):
        assert detector.detect_batch([]) == []


class TestCensor:
    def test_black_style(self, detector, blank_image_path, tmp_path):
        out = str(tmp_path / "out_black.jpg")
        result = detector.censor(blank_image_path, output_path=out, censor_style="black")
        assert os.path.isfile(result)

    def test_blur_style(self, detector, blank_image_path, tmp_path):
        out = str(tmp_path / "out_blur.jpg")
        result = detector.censor(blank_image_path, output_path=out, censor_style="blur")
        assert os.path.isfile(result)

    def test_pixel_style(self, detector, blank_image_path, tmp_path):
        out = str(tmp_path / "out_pixel.jpg")
        result = detector.censor(blank_image_path, output_path=out, censor_style="pixel")
        assert os.path.isfile(result)

    def test_ndarray_input(self, detector, blank_ndarray, tmp_path):
        out = str(tmp_path / "out_array.jpg")
        result = detector.censor(blank_ndarray, output_path=out)
        assert os.path.isfile(result)

    def test_bytes_input(self, detector, blank_image_path, tmp_path):
        out = str(tmp_path / "out_bytes.jpg")
        with open(blank_image_path, "rb") as f:
            data = f.read()
        result = detector.censor(data, output_path=out)
        assert os.path.isfile(result)

    def test_invalid_style_raises(self, detector, blank_image_path, tmp_path):
        with pytest.raises(ValueError, match="censor_style"):
            detector.censor(blank_image_path, censor_style="rainbow")

    def test_auto_output_path(self, detector, blank_image_path):
        result = detector.censor(blank_image_path)
        assert os.path.isfile(result)
        os.remove(result)   # cleanup


class TestLabels:
    def test_labels_count(self):
        assert len(LABELS) == 18

    def test_labels_are_strings(self):
        assert all(isinstance(l, str) for l in LABELS)

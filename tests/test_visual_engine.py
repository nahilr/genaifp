"""Tests for visual_engine module."""

import numpy as np
import pytest
from PIL import Image

from genaifp.visual_engine import VisualEngine, extract_visual_features


@pytest.fixture(scope="module")
def engine():
    return VisualEngine()


class TestVisualEngine:
    def test_output_dimension(self, engine):
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        feat = engine.extract(arr)
        assert feat.shape == (384,)
        assert feat.dtype == np.float32

    def test_l2_normalized(self, engine):
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        feat = engine.extract(arr)
        norm = np.linalg.norm(feat)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_pil_input(self, engine):
        pil_img = Image.new("RGB", (300, 300), color=(128, 64, 32))
        feat = engine.extract(pil_img)
        assert feat.shape == (384,)

    def test_batch_extraction(self, engine):
        batch = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(3)]
        feats = engine.extract_batch(batch)
        assert feats.shape == (3, 384)
        for i in range(3):
            assert np.isclose(np.linalg.norm(feats[i]), 1.0, atol=1e-5)

    def test_convenience_function(self):
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feat = extract_visual_features(arr)
        assert feat.shape == (384,)

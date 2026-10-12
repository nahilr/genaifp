"""Tests for robustness_bench module."""

import numpy as np
import pytest

from genaifp.robustness_bench import (
    PERTURBATIONS,
    apply_downscale_upscale,
    apply_gaussian_blur,
    apply_jpeg,
)


class TestPerturbations:
    @pytest.fixture
    def test_image(self):
        # 512x512 RGB uint8
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)

    def test_apply_jpeg(self, test_image):
        compressed = apply_jpeg(test_image, quality=85)
        assert compressed.shape == (512, 512, 3)
        assert compressed.dtype == np.uint8
        # Should not be identical to uncompressed random noise
        assert not np.array_equal(compressed, test_image)

    def test_apply_gaussian_blur(self, test_image):
        blurred = apply_gaussian_blur(test_image, sigma=1.0)
        assert blurred.shape == (512, 512, 3)
        assert blurred.dtype == np.uint8
        # Variance should decrease with blur
        assert blurred.std() < test_image.std()

    def test_apply_downscale_upscale(self, test_image):
        restored = apply_downscale_upscale(test_image, scale=0.75)
        assert restored.shape == (512, 512, 3)
        assert restored.dtype == np.uint8

    def test_perturbations_dict(self):
        assert "jpeg" in PERTURBATIONS
        assert "blur" in PERTURBATIONS
        assert "downscale" in PERTURBATIONS

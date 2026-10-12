"""Tests for preprocessor module."""

import tempfile

import numpy as np
import pytest
from PIL import Image

from genaifp import (
    CROP_SIZE,
    SRM_KERNEL,
    Y_COEFFS,
    apply_srm_highpass,
    load_and_crop,
    preprocess_image,
    rgb_to_luminance,
)


class TestLoadAndCrop:
    def test_center_crop_512_from_1024(self):
        """Center crop 1024x1024 -> 512x512."""
        img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.fromarray(img).save(f.name)
            cropped = load_and_crop(f.name, 512)

        assert cropped.shape == (512, 512, 3)
        assert cropped.dtype == np.float64

    def test_raises_on_small_image(self):
        """Should raise if image smaller than crop size."""
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.fromarray(img).save(f.name)
            with pytest.raises(ValueError):
                load_and_crop(f.name, 512)

    def test_non_square_image(self):
        """Handle non-square images (e.g., 768x512)."""
        img = np.random.randint(0, 255, (768, 512, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.fromarray(img).save(f.name)
            cropped = load_and_crop(f.name, 512)

        assert cropped.shape == (512, 512, 3)


class TestRGBToLuminance:
    def test_shape(self):
        """Output shape should be (H, W)."""
        rgb = np.random.rand(512, 512, 3)
        y = rgb_to_luminance(rgb)
        assert y.shape == (512, 512)

    def test_coefficients(self):
        """Verify ITU-R BT.601 coefficients."""
        # Pure red -> Y = 0.299
        rgb = np.zeros((10, 10, 3))
        rgb[:, :, 0] = 1.0
        y = rgb_to_luminance(rgb)
        assert np.allclose(y, 0.299)

        # Pure green -> Y = 0.587
        rgb = np.zeros((10, 10, 3))
        rgb[:, :, 1] = 1.0
        y = rgb_to_luminance(rgb)
        assert np.allclose(y, 0.587)

        # Pure blue -> Y = 0.114
        rgb = np.zeros((10, 10, 3))
        rgb[:, :, 2] = 1.0
        y = rgb_to_luminance(rgb)
        assert np.allclose(y, 0.114)


class TestApplySRMHighpass:
    def test_shape_preserved(self):
        """Output shape matches input."""
        y = np.random.rand(512, 512)
        residual = apply_srm_highpass(y)
        assert residual.shape == (512, 512)
        assert residual.dtype == np.float32

    def test_zero_mean_on_flat(self):
        """SRM on flat field should give near-zero."""
        y = np.ones((512, 512)) * 128
        residual = apply_srm_highpass(y)
        assert np.abs(residual.mean()) < 1e-5

    def test_srm_kernel_values(self):
        """Verify kernel is correct."""
        expected = np.array(
            [
                [-1, 2, -1],
                [2, -4, 2],
                [-1, 2, -1],
            ],
            dtype=np.float64,
        )
        assert np.array_equal(SRM_KERNEL, expected)


class TestPreprocessImage:
    def test_end_to_end(self):
        """Full pipeline produces correct output."""
        img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.fromarray(img).save(f.name)
            residual = preprocess_image(f.name)

        assert residual.shape == (512, 512)
        assert residual.dtype == np.float32

    def test_deterministic(self):
        """Identical input produces identical output."""
        img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.fromarray(img).save(f.name)
            r1 = preprocess_image(f.name)
            r2 = preprocess_image(f.name)

        assert np.allclose(r1, r2, atol=1e-6)


class TestConstants:
    def test_crop_size(self):
        assert CROP_SIZE == 512

    def test_y_coeffs_sum_to_one(self):
        assert np.isclose(Y_COEFFS.sum(), 1.0)

    def test_srm_kernel_sum_zero(self):
        """High-pass kernel should sum to zero."""
        assert np.isclose(SRM_KERNEL.sum(), 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

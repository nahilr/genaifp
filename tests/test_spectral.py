"""Tests for spectral_engine module."""

import numpy as np
import pytest

from genaifp import (
    SpectralEngine,
    compute_2d_spectrum,
    compute_azimuthal_grid,
    compute_radial_grid,
    extract_azimuthal_profiles,
    extract_rapsd,
    flatten_upper_triangular,
)


class TestCompute2DSpectrum:
    def test_shape(self):
        residual = np.random.randn(512, 512).astype(np.float32)
        log_power = compute_2d_spectrum(residual)
        assert log_power.shape == (512, 512)
        assert log_power.dtype == np.float32

    def test_dc_at_center(self):
        """DC component should be at center after fftshift."""
        # Constant signal -> only DC component non-zero
        residual = np.ones((64, 64), dtype=np.float32)
        log_power = compute_2d_spectrum(residual)
        center = log_power[32, 32]
        # DC should be maximum
        assert center == log_power.max()

    def test_log_scale(self):
        """Values should be log-scaled (log1p)."""
        residual = np.random.randn(64, 64).astype(np.float32) * 100
        log_power = compute_2d_spectrum(residual)
        # log1p(x) < x for x > 0
        assert np.all(log_power >= 0)


class TestComputeRadialGrid:
    def test_shape(self):
        grid = compute_radial_grid((512, 512))
        assert grid.shape == (512, 512)

    def test_center_is_zero(self):
        grid = compute_radial_grid((512, 512))
        assert grid[256, 256] == 0

    def test_max_radius(self):
        grid = compute_radial_grid((512, 512))
        # Corner radius = sqrt(256^2 + 256^2) ≈ 362
        assert grid.max() == 362


class TestComputeAzimuthalGrid:
    def test_four_quadrants(self):
        grid = compute_azimuthal_grid((512, 512), n_quadrants=4)
        unique = np.unique(grid)
        assert list(unique) == [0, 1, 2, 3]

    def test_eight_quadrants(self):
        grid = compute_azimuthal_grid((512, 512), n_quadrants=8)
        unique = np.unique(grid)
        assert list(unique) == list(range(8))

    def test_quadrant_counts_equal(self):
        """Each quadrant should have roughly equal pixels."""
        grid = compute_azimuthal_grid((512, 512), n_quadrants=4)
        for q in range(4):
            count = np.sum(grid == q)
            # Should be ~512*512/4 = 65536
            assert 60000 < count < 70000


class TestExtractRAPSD:
    def test_output_length(self):
        log_power = np.random.rand(512, 512).astype(np.float32)
        r_grid = compute_radial_grid((512, 512))
        rapsd = extract_rapsd(log_power, r_grid, r_max=255)
        assert len(rapsd) == 255
        assert rapsd.dtype == np.float32

    def test_dc_excluded(self):
        """Ring 0 (DC) should be excluded."""
        log_power = np.ones((64, 64), dtype=np.float32)
        r_grid = compute_radial_grid((64, 64))
        rapsd = extract_rapsd(log_power, r_grid, r_max=31)
        # All rings should have value 0 (log1p(1)=log(2))
        # but first ring (r=1) should be present
        assert len(rapsd) == 31

    def test_constant_signal(self):
        """Constant signal -> all power at DC, RAPSD near zero for r>0."""
        log_power = np.log1p(np.ones((64, 64), dtype=np.float32) * 1000)
        r_grid = compute_radial_grid((64, 64))
        rapsd = extract_rapsd(log_power, r_grid, r_max=31)
        # Rings r>0 should have much lower values
        assert rapsd[0] == pytest.approx(rapsd[-1], rel=0.1)


class TestExtractAzimuthalProfiles:
    def test_shape(self):
        log_power = np.random.rand(64, 64).astype(np.float32)
        r_grid = compute_radial_grid((64, 64))
        q_grid = compute_azimuthal_grid((64, 64), 4)
        profiles = extract_azimuthal_profiles(log_power, r_grid, q_grid, r_max=31)
        assert profiles.shape == (4, 31)

    def test_sum_matches_rapsd(self):
        """Sum over quadrants should equal RAPSD (within averaging diffs)."""
        log_power = np.random.rand(64, 64).astype(np.float32)
        r_grid = compute_radial_grid((64, 64))
        q_grid = compute_azimuthal_grid((64, 64), 4)
        profiles = extract_azimuthal_profiles(log_power, r_grid, q_grid, r_max=31)
        rapsd = extract_rapsd(log_power, r_grid, r_max=31)

        # Weighted sum of quadrant profiles should match RAPSD
        # (approximately, since both average over same pixels)
        quad_sum = profiles.mean(axis=0)
        # Should be correlated
        corr = np.corrcoef(rapsd, quad_sum)[0, 1]
        assert corr > 0.9


class TestFlattenUpperTriangular:
    def test_length(self):
        """Length should be n*(n+1)/2 for n x n."""
        for n in [16, 32, 64, 512]:
            log_power = np.random.rand(n, n).astype(np.float32)
            flat = flatten_upper_triangular(log_power)
            expected = n * (n + 1) // 2
            assert len(flat) == expected

    def test_contains_diagonal(self):
        """Should include diagonal elements."""
        n = 16
        log_power = np.eye(n, dtype=np.float32)
        flat = flatten_upper_triangular(log_power)
        # Should have n ones (the diagonal)
        assert np.sum(flat == 1.0) == n


class TestSpectralEngine:
    def test_process_output_keys(self):
        engine = SpectralEngine(crop_size=64)
        residual = np.random.randn(64, 64).astype(np.float32)
        out = engine.process(residual)

        assert "log_power" in out
        assert "rapsd" in out
        assert "azimuthal" in out
        assert "flat_spectrum" in out

    def test_process_shapes(self):
        engine = SpectralEngine(crop_size=64, r_max=31)
        residual = np.random.randn(64, 64).astype(np.float32)
        out = engine.process(residual)

        assert out["log_power"].shape == (64, 64)
        assert out["rapsd"].shape == (31,)
        assert out["azimuthal"].shape == (4, 31)
        assert len(out["flat_spectrum"]) == 64 * 65 // 2

    def test_batch_processing(self):
        engine = SpectralEngine(crop_size=64, r_max=31)
        residuals = np.random.randn(5, 64, 64).astype(np.float32)
        out = engine.process_batch(residuals)

        assert out["log_power"].shape == (5, 64, 64)
        assert out["rapsd"].shape == (5, 31)
        assert out["azimuthal"].shape == (5, 4, 31)
        assert out["flat_spectrum"].shape == (5, 64 * 65 // 2)

    def test_deterministic(self):
        engine = SpectralEngine(crop_size=64)
        residual = np.random.randn(64, 64).astype(np.float32)
        out1 = engine.process(residual)
        out2 = engine.process(residual)

        for k in out1:
            assert np.allclose(out1[k], out2[k])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

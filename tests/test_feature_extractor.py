"""Tests for feature_extractor module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from genaifp.feature_extractor import FEATURE_NAMES_12, FeatureExtractor


class TestFeatureExtractor:
    def test_feature_names_length(self):
        assert len(FEATURE_NAMES_12) == 12
        assert "pasr" in FEATURE_NAMES_12
        assert "alpha" in FEATURE_NAMES_12
        assert "directional_var" in FEATURE_NAMES_12

    def test_extract_12_scalar_shape(self):
        r_max = 255
        rapsd = np.abs(np.random.randn(r_max)).astype(np.float32) + 0.1
        azimuthal = np.abs(np.random.randn(4, r_max)).astype(np.float32) + 0.1

        feats = FeatureExtractor.extract_12_scalar(rapsd, azimuthal, n_subbands=8)
        assert feats.shape == (12,)
        assert feats.dtype == np.float32
        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()

    def test_extract_12_scalar_zeros(self):
        rapsd = np.zeros(255, dtype=np.float32)
        azimuthal = np.zeros((4, 255), dtype=np.float32)
        feats = FeatureExtractor.extract_12_scalar(rapsd, azimuthal)
        assert feats.shape == (12,)
        assert np.allclose(feats, 0)

    def test_fit_and_transform_pca(self):
        N = 50
        rng = np.random.default_rng(42)
        # Use small D for fast unit testing
        D_small = 1000
        flat = rng.normal(size=(N, D_small)).astype(np.float32)

        extractor = FeatureExtractor(n_pca_components=16)
        extractor.fit_pca(flat)
        assert extractor._fitted

        # Single transform
        proj_single = extractor.transform_pca(flat[0])
        assert proj_single.shape == (16,)

        # Batch transform
        proj_batch = extractor.transform_pca(flat[:5])
        assert proj_batch.shape == (5, 16)
        assert np.allclose(proj_batch[0], proj_single)

    def test_save_and_load(self):
        N = 30
        D = 200
        flat = np.random.randn(N, D).astype(np.float32)

        extractor = FeatureExtractor(n_pca_components=8)
        extractor.fit_pca(flat)
        sample_out = extractor.transform_pca(flat[0])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            save_path = f.name
            extractor.save(save_path)

            loaded = FeatureExtractor.load(save_path)
            assert loaded._fitted
            loaded_out = loaded.transform_pca(flat[0])
            assert np.allclose(sample_out, loaded_out, atol=1e-6)

        Path(save_path).unlink(missing_ok=True)

    def test_unfitted_raises(self):
        extractor = FeatureExtractor()
        with pytest.raises(RuntimeError):
            extractor.transform_pca(np.random.randn(100).astype(np.float32))
        with pytest.raises(RuntimeError):
            extractor.save("/tmp/fail.pkl")

    def test_extract_hybrid(self):
        extractor = FeatureExtractor(n_pca_components=16)
        N = 30
        D = 100
        flat = np.random.randn(N, D).astype(np.float32)
        extractor.fit_pca(flat)

        spectral_out = {
            "flat_spectrum": flat[0],
            "rapsd": np.abs(np.random.randn(255)).astype(np.float32),
            "azimuthal": np.abs(np.random.randn(4, 255)).astype(np.float32),
        }
        mock_vis = np.random.randn(384).astype(np.float32)

        res = extractor.extract_hybrid(spectral_out, visual_features=mock_vis)
        assert "hybrid_features" in res
        assert "visual_features" in res
        assert "pca_features" in res
        assert "scalar_12" in res
        # Dim = 384 (vis) + 16 (pca) + 12 (scalar) = 412
        assert res["hybrid_features"].shape == (384 + 16 + 12,)
        assert np.allclose(res["hybrid_features"][:384], mock_vis)

    def test_extract_requires_fitted_pca(self):
        extractor = FeatureExtractor(n_pca_components=8)
        with pytest.raises(RuntimeError, match="PCA is not fitted"):
            extractor.extract(
                {
                    "flat_spectrum": np.ones(16, dtype=np.float32),
                    "rapsd": np.ones(8, dtype=np.float32),
                    "azimuthal": np.ones((4, 8), dtype=np.float32),
                }
            )

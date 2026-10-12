"""Module 3: Feature Extraction.

Two tracks:
- Track A (primary): PCA on flattened upper-triangular log-spectrum
- Track B (fallback/interp): 12 hand-crafted scalars from RAPSD + azimuthal
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 12-scalar feature indices (for reference)
FEATURE_NAMES_12 = [
    "pasr",  # Peak-to-Average Spectral Ratio
    "alpha",  # High-frequency spectral slope (decay exponent)
    "beta",  # Intercept of log-log fit
    "E_1",
    "E_2",
    "E_3",
    "E_4",  # Sub-band energies (8 bands)
    "E_5",
    "E_6",
    "E_7",
    "E_8",
    "directional_var",  # Inter-quadrant variance
]

VISUAL_FEATURE_DIM = 384
SPECTRAL_PCA_DIM = 128
SPECTRAL_SCALAR_DIM = 12
HYBRID_FEATURE_DIM = VISUAL_FEATURE_DIM + SPECTRAL_PCA_DIM + SPECTRAL_SCALAR_DIM  # 524


class FeatureExtractor:
    """Extract features from spectral engine output.

    Maintains fitted PCA + StandardScaler for consistent projection.
    """

    def __init__(self, n_pca_components: int = 128, n_subbands: int = 8, r_max: int = 255):
        self.n_pca_components = n_pca_components
        self.n_subbands = n_subbands
        self.r_max = r_max

        # PCA pipeline (fitted on training data)
        self.pca: PCA | None = None
        self.scaler: StandardScaler | None = None
        self._fitted = False

    def fit_pca(self, flat_spectra: np.ndarray) -> FeatureExtractor:
        """Fit PCA + scaler on training flat spectra.

        Args:
            flat_spectra: (N, D) array from spectral_engine['flat_spectrum']
        """
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(flat_spectra)

        n_comp = min(self.n_pca_components, scaled.shape[1], scaled.shape[0] - 1)
        self.pca = PCA(n_components=n_comp, random_state=42, svd_solver="randomized")
        self.pca.fit(scaled)

        self._fitted = True
        return self

    def transform_pca(self, flat_spectra: np.ndarray) -> np.ndarray:
        """Project flat spectra to PCA space.

        Args:
            flat_spectra: (N, D) or (D,) array

        Returns:
            (N, n_components) or (n_components,) PCA features
        """
        if not self._fitted:
            raise RuntimeError("PCA not fitted. Call fit_pca() first.")

        single = flat_spectra.ndim == 1
        if single:
            flat_spectra = flat_spectra[None, :]

        scaled = self.scaler.transform(flat_spectra)
        pca_feats = self.pca.transform(scaled)

        return pca_feats[0] if single else pca_feats

    @staticmethod
    def extract_12_scalar(
        rapsd: np.ndarray, azimuthal: np.ndarray, n_subbands: int = 8
    ) -> np.ndarray:
        """Extract the 12 hand-crafted scalar features.

        Args:
            rapsd: (r_max,) radially averaged log-power
            azimuthal: (n_quadrants, r_max) directional profiles
            n_subbands: number of equal-width radial bands

        Returns:
            (12,) feature vector
        """
        # Normalize RAPSD to unit sum
        P = rapsd.astype(np.float64)
        P_sum = P.sum()
        if P_sum == 0:
            return np.zeros(12, dtype=np.float32)
        Pn = P / P_sum

        r = np.arange(1, len(Pn) + 1, dtype=np.float64)

        # 1. PASR
        pasr = float(Pn.max() / Pn.mean())

        # 2-3. Spectral slope (alpha) and intercept (beta) on upper 50% of frequencies
        # Use region where spectrum decays (r >= peak radius)
        peak_r = int(np.argmax(Pn[: len(Pn) // 2]))  # peak in lower half
        fit_start = max(peak_r, len(Pn) // 2)  # ensure upper half
        fit_start = min(fit_start, len(Pn) - 10)  # need enough points

        if fit_start < len(Pn) - 5:
            mask = r >= r[fit_start]
            log_r = np.log(r[mask])
            log_P = np.log(Pn[mask] + 1e-18)
            if len(log_r) >= 3:
                slope, intercept = np.polyfit(log_r, log_P, 1)
                alpha = float(-slope)  # spec: log P = -alpha log r + beta
                beta = float(intercept)
            else:
                alpha, beta = 0.0, 0.0
        else:
            alpha, beta = 0.0, 0.0

        # 4-11. Sub-band energies (8 equal-width bands)
        edges = np.linspace(1, len(Pn) + 1, n_subbands + 1).astype(int)
        E = np.array(
            [float(Pn[edges[i] - 1 : edges[i + 1] - 1].sum()) for i in range(n_subbands)],
            dtype=np.float32,
        )

        # 12. Directional variance
        # azimuthal shape: (n_quadrants, r_max) - already mean per ring per quadrant
        # Normalize each quadrant's profile to unit sum, then compute variance across quadrants
        A = azimuthal.astype(np.float64)
        A_sum = A.sum(axis=1, keepdims=True)
        A_norm = np.where(A_sum > 0, A / A_sum, 0)
        # Variance across quadrants at each radius, then average over radii
        dir_var = float(np.mean(np.var(A_norm, axis=0)))

        # Assemble 12-vector
        features = np.array([pasr, alpha, beta] + E.tolist() + [dir_var], dtype=np.float32)
        return features

    def extract(self, spectral_out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Extract both feature tracks from spectral engine output.

        Args:
            spectral_out: dict from SpectralEngine.process()

        Returns:
            dict with:
                - 'pca_features': (n_components,) or (N, n_components)
                - 'scalar_12': (12,) or (N, 12)
                - 'flat_spectrum': original flattened spectrum (for debugging)
        """
        if not self._fitted:
            raise RuntimeError("PCA is not fitted. Fit or load the model's feature extractor first.")
        flat = spectral_out["flat_spectrum"]
        rapsd = spectral_out["rapsd"]
        azimuthal = spectral_out["azimuthal"]

        single = flat.ndim == 1
        if single:
            flat = flat[None, :]
            rapsd = rapsd[None, :]
            azimuthal = azimuthal[None, :, :]

        N = flat.shape[0]

        # Track A: PCA features
        pca_feats = self.transform_pca(flat)

        # Track B: 12-scalar features
        scalar_feats = np.zeros((N, 12), dtype=np.float32)
        for i in range(N):
            scalar_feats[i] = self.extract_12_scalar(rapsd[i], azimuthal[i], self.n_subbands)

        if single:
            return {
                "pca_features": pca_feats[0],
                "scalar_12": scalar_feats[0],
                "flat_spectrum": flat[0],
            }
        else:
            return {
                "pca_features": pca_feats,
                "scalar_12": scalar_feats,
                "flat_spectrum": flat,
            }

    def extract_hybrid(
        self,
        spectral_out: dict[str, np.ndarray],
        visual_features: np.ndarray | None = None,
        image: Any | None = None,
        visual_engine: Any | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract fused visual + spectral features (Track A + Track B).

        Fused vector: [visual_embedding (384), pca_spectral (128), scalar_12 (12)] = 524 dims.
        """
        spectral_feats = self.extract(spectral_out)
        pca_feats = spectral_feats["pca_features"]
        scalar_feats = spectral_feats["scalar_12"]

        if visual_features is None:
            if image is None:
                raise ValueError("Must provide either visual_features or image to extract_hybrid")
            from .visual_engine import extract_visual_features

            if visual_engine is not None:
                visual_features = visual_engine.extract(image)
            else:
                visual_features = extract_visual_features(image)

        single = pca_feats.ndim == 1
        if single:
            hybrid = np.concatenate([visual_features, pca_feats, scalar_feats], axis=0).astype(
                np.float32
            )
        else:
            hybrid = np.concatenate([visual_features, pca_feats, scalar_feats], axis=1).astype(
                np.float32
            )

        res = dict(spectral_feats)
        res["visual_features"] = visual_features
        res["hybrid_features"] = hybrid
        return res

    def save(self, path: str | Path) -> None:
        """Save fitted PCA + scaler to disk."""
        if not self._fitted:
            raise RuntimeError("Nothing to save - PCA not fitted")
        joblib.dump(
            {
                "pca": self.pca,
                "scaler": self.scaler,
                "n_pca_components": self.n_pca_components,
                "n_subbands": self.n_subbands,
                "r_max": self.r_max,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> FeatureExtractor:
        """Load fitted feature extractor from disk."""
        data = joblib.load(path)
        extractor = cls(
            n_pca_components=data["n_pca_components"],
            n_subbands=data["n_subbands"],
            r_max=data["r_max"],
        )
        extractor.pca = data["pca"]
        extractor.scaler = data["scaler"]
        extractor._fitted = True
        return extractor


if __name__ == "__main__":
    # Self-test
    import time

    # Mock spectral output
    N = 100
    D = 512 * 513 // 2  # upper triangular of 512x512
    flat_spectra = np.random.randn(N, D).astype(np.float32)

    extractor = FeatureExtractor(n_pca_components=128)
    extractor.fit_pca(flat_spectra)

    # Mock rapsd/azimuthal
    r_max = 255
    rapsd = np.abs(np.random.randn(N, r_max)).astype(np.float32) + 1
    azimuthal = np.abs(np.random.randn(N, 4, r_max)).astype(np.float32) + 1

    spectral_out = {
        "flat_spectrum": flat_spectra[:5],
        "rapsd": rapsd[:5],
        "azimuthal": azimuthal[:5],
    }

    t0 = time.perf_counter()
    feats = extractor.extract(spectral_out)
    elapsed = time.perf_counter() - t0

    print(f"PCA features shape: {feats['pca_features'].shape}")
    print(f"12-scalar shape: {feats['scalar_12'].shape}")
    print(f"Feature names: {FEATURE_NAMES_12}")
    print(f"Sample 12-scalar: {feats['scalar_12'][0]}")
    print(f"Extract time: {elapsed * 1000:.1f} ms for 5 samples")
    print(f"PCA explained variance ratio (top 10): {extractor.pca.explained_variance_ratio_[:10]}")
    print(f"Total explained variance: {extractor.pca.explained_variance_ratio_.sum():.4f}")

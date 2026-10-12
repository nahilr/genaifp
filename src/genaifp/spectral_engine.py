"""Module 2: 2D Spectral Decomposition.

Computes 2D FFT of residual, zero-centered power spectrum, and provides:
- Full log-magnitude spectrum (for PCA projection)
- RAPSD + azimuthal profiles (for 12-scalar fallback features)
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, fftshift


def compute_2d_spectrum(residual: np.ndarray) -> np.ndarray:
    """Compute zero-centered 2D log-power spectrum.

    Args:
        residual: float32 array (H, W) from preprocessor

    Returns:
        log|F|^2 array (H, W) with DC component at center
    """
    # 2D FFT
    F = fft2(residual)
    # Shift zero-frequency to center
    F_shifted = fftshift(F)
    # Power spectrum |F|^2
    power = np.abs(F_shifted) ** 2
    # Log scale for dynamic range
    log_power = np.log1p(power).astype(np.float32)
    return log_power


def compute_radial_grid(shape: tuple[int, int]) -> np.ndarray:
    """Compute integer radial coordinate grid centered at (H/2, W/2).

    Returns:
        r_grid: int array (H, W) where r = floor(sqrt((y-cy)^2 + (x-cx)^2))
    """
    H, W = shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return np.floor(r).astype(np.int32)


def compute_azimuthal_grid(shape: tuple[int, int], n_quadrants: int = 4) -> np.ndarray:
    """Compute azimuthal quadrant labels (0 to n_quadrants-1).

    Args:
        shape: (H, W)
        n_quadrants: number of equal-angle sectors (default 4 = 90° each)

    Returns:
        q_grid: int array (H, W) with values 0..n_quadrants-1
    """
    H, W = shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    # Angle in degrees [0, 360)
    angle = np.degrees(np.arctan2(y - cy, x - cx)) + 180
    # Bin into equal sectors
    q = (angle // (360 / n_quadrants)).astype(np.int32)
    return np.clip(q, 0, n_quadrants - 1)


def extract_rapsd(
    log_power: np.ndarray, r_grid: np.ndarray, r_max: int | None = None
) -> np.ndarray:
    """Extract Radially Averaged Power Spectral Density (RAPSD).

    Args:
        log_power: log|F|^2 array (H, W)
        r_grid: radial coordinate grid from compute_radial_grid
        r_max: maximum radius to include (default: min(H,W)//2 - 1)

    Returns:
        rapsd: float32 array (r_max,) - mean power per radial ring
    """
    H, W = log_power.shape
    if r_max is None:
        r_max = min(H, W) // 2 - 1  # discard sparse outer ring

    # Flatten
    r_flat = r_grid.ravel()
    p_flat = log_power.ravel()

    # Only keep valid rings 1..r_max (ring 0 is DC)
    valid = (r_flat >= 1) & (r_flat <= r_max)
    r_flat = r_flat[valid]
    p_flat = p_flat[valid]

    # Bin by radius
    counts = np.bincount(r_flat, minlength=r_max + 1)[1 : r_max + 1]
    sums = np.bincount(r_flat, weights=p_flat, minlength=r_max + 1)[1 : r_max + 1]

    # Avoid division by zero
    rapsd = np.zeros(r_max, dtype=np.float32)
    mask = counts > 0
    rapsd[mask] = sums[mask] / counts[mask]

    return rapsd


def extract_azimuthal_profiles(
    log_power: np.ndarray,
    r_grid: np.ndarray,
    q_grid: np.ndarray,
    r_max: int | None = None,
    n_quadrants: int = 4,
) -> np.ndarray:
    """Extract directional profiles per quadrant.

    Args:
        log_power: log|F|^2 array (H, W)
        r_grid: radial coordinate grid
        q_grid: quadrant grid from compute_azimuthal_grid
        r_max: maximum radius
        n_quadrants: number of sectors

    Returns:
        profiles: float32 array (n_quadrants, r_max) - mean power per ring per quadrant
    """
    H, W = log_power.shape
    if r_max is None:
        r_max = min(H, W) // 2 - 1

    r_flat = r_grid.ravel()
    q_flat = q_grid.ravel()
    p_flat = log_power.ravel()

    valid = (r_flat >= 1) & (r_flat <= r_max)
    r_flat = r_flat[valid]
    q_flat = q_flat[valid]
    p_flat = p_flat[valid]

    profiles = np.zeros((n_quadrants, r_max), dtype=np.float32)

    for q in range(n_quadrants):
        qmask = q_flat == q
        if not qmask.any():
            continue
        rq = r_flat[qmask]
        pq = p_flat[qmask]
        counts = np.bincount(rq, minlength=r_max + 1)[1 : r_max + 1]
        sums = np.bincount(rq, weights=pq, minlength=r_max + 1)[1 : r_max + 1]
        m = counts > 0
        profiles[q, m] = sums[m] / counts[m]

    return profiles


def flatten_upper_triangular(log_power: np.ndarray) -> np.ndarray:
    """Flatten upper-triangular part of log-power spectrum for PCA.

    The spectrum is Hermitian-symmetric, so lower triangle is redundant.
    Returns upper triangle including diagonal, row-major order.
    """
    H, W = log_power.shape
    # Upper triangular mask
    mask = np.triu(np.ones((H, W), dtype=bool))
    return log_power[mask].astype(np.float32)  # length H*(H+1)/2 for square


class SpectralEngine:
    """Stateful engine for repeated spectral analysis.

    Precomputes radial/azimuthal grids to avoid redundant allocation.
    """

    def __init__(self, crop_size: int = 512, r_max: int | None = None, n_quadrants: int = 4):
        self.crop_size = crop_size
        self.r_max = r_max or (crop_size // 2 - 1)
        self.n_quadrants = n_quadrants

        # Precompute grids
        self.r_grid = compute_radial_grid((crop_size, crop_size))
        self.q_grid = compute_azimuthal_grid((crop_size, crop_size), n_quadrants)

    def process(self, residual: np.ndarray) -> dict[str, np.ndarray]:
        """Full spectral processing pipeline.

        Returns dict with:
            - 'log_power': (H, W) full log-magnitude spectrum
            - 'rapsd': (r_max,) radially averaged profile
            - 'azimuthal': (n_quadrants, r_max) directional profiles
            - 'flat_spectrum': (H*(H+1)/2,) flattened upper triangle for PCA
        """
        log_power = compute_2d_spectrum(residual)
        rapsd = extract_rapsd(log_power, self.r_grid, self.r_max)
        azimuthal = extract_azimuthal_profiles(
            log_power, self.r_grid, self.q_grid, self.r_max, self.n_quadrants
        )
        flat = flatten_upper_triangular(log_power)

        return {
            "log_power": log_power,
            "rapsd": rapsd,
            "azimuthal": azimuthal,
            "flat_spectrum": flat,
        }

    def process_batch(self, residuals: np.ndarray) -> dict[str, np.ndarray]:
        """Process batch of residuals.

        Args:
            residuals: (N, H, W) float32 array

        Returns:
            dict with batched arrays
        """
        N = residuals.shape[0]
        log_powers = np.zeros((N, self.crop_size, self.crop_size), dtype=np.float32)
        rapsds = np.zeros((N, self.r_max), dtype=np.float32)
        azimuthals = np.zeros((N, self.n_quadrants, self.r_max), dtype=np.float32)
        flats = np.zeros((N, self.crop_size * (self.crop_size + 1) // 2), dtype=np.float32)

        for i in range(N):
            out = self.process(residuals[i])
            log_powers[i] = out["log_power"]
            rapsds[i] = out["rapsd"]
            azimuthals[i] = out["azimuthal"]
            flats[i] = out["flat_spectrum"]

        return {
            "log_power": log_powers,
            "rapsd": rapsds,
            "azimuthal": azimuthals,
            "flat_spectrum": flats,
        }


if __name__ == "__main__":
    # Self-test with synthetic data
    import time

    # Synthetic residual-like signal (1/f noise)
    rng = np.random.default_rng(0)
    H = W = 512
    y, x = np.ogrid[:H, :W]
    cx, cy = W // 2, H // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 1
    F = rng.normal(size=(H, W)) / (r**0.75)
    residual = np.real(np.fft.ifft2(F)).astype(np.float32)

    engine = SpectralEngine(crop_size=512)

    # Benchmark
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = engine.process(residual)
        times.append(time.perf_counter() - t0)

    print(f"log_power shape: {out['log_power'].shape}")
    print(f"rapsd shape: {out['rapsd'].shape}")
    print(f"azimuthal shape: {out['azimuthal'].shape}")
    print(f"flat_spectrum shape: {out['flat_spectrum'].shape}")
    print(f"Mean time: {np.mean(times) * 1000:.1f} ms")
    print(f"RAPSD range: [{out['rapsd'].min():.3f}, {out['rapsd'].max():.3f}]")
    print(f"Azimuthal row sums: {out['azimuthal'].sum(axis=1)}")

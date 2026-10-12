"""Image preprocessing and visual/spatial-frequency feature extraction."""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy.ndimage import convolve
from scipy.optimize import minimize_scalar
from scipy.fft import fft2, fftshift

from config import CROP_SIZE, LUMA, SRM_KERNEL


def center_crop(rgb: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    if rgb.ndim == 3 and rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    rgb = rgb[:, :, :3]
    height, width = rgb.shape[:2]
    pad_y, pad_x = max(0, size - height), max(0, size - width)
    if pad_y or pad_x:
        top, left = pad_y // 2, pad_x // 2
        rgb = np.pad(
            rgb,
            ((top, pad_y - top), (left, pad_x - left), (0, 0)),
            mode="reflect" if min(height, width) > 1 else "edge",
        )
        height, width = rgb.shape[:2]
    top, left = (height - size) // 2, (width - size) // 2
    return rgb[top : top + size, left : left + size]

def build_spectral_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.ogrid[:CROP_SIZE, :CROP_SIZE]
    radius = np.floor(np.sqrt((y - CROP_SIZE // 2) ** 2 + (x - CROP_SIZE // 2) ** 2)).astype(np.int32)
    angle = np.degrees(np.arctan2(y - CROP_SIZE // 2, x - CROP_SIZE // 2)) + 180
    quadrant = np.clip((angle // 90).astype(np.int32), 0, 3)
    triangle = np.triu_indices(CROP_SIZE)
    return radius, quadrant, triangle

def spectral_features(rgb: np.ndarray, grids: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    radius_grid, quadrant_grid, triangle = grids
    luminance = np.dot(rgb.astype(np.float64), LUMA)
    residual = convolve(luminance, SRM_KERNEL, mode="reflect").astype(np.float32)
    shifted = fftshift(fft2(residual))
    log_power = np.log1p(np.abs(shifted) ** 2).astype(np.float32)
    radius = radius_grid.ravel()
    quadrant = quadrant_grid.ravel()
    power = log_power.ravel()
    r_max = CROP_SIZE // 2 - 1
    valid = (radius >= 1) & (radius <= r_max)
    radial = radius[valid]
    sector = quadrant[valid]
    values = power[valid]
    counts = np.bincount(radial, minlength=r_max + 1)[1 : r_max + 1]
    sums = np.bincount(radial, weights=values, minlength=r_max + 1)[1 : r_max + 1]
    rapsd = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float64), where=counts > 0)
    azimuthal = np.zeros((4, r_max), dtype=np.float64)
    for sector_id in range(4):
        selected = sector == sector_id
        bins = radial[selected]
        p = values[selected]
        count = np.bincount(bins, minlength=r_max + 1)[1 : r_max + 1]
        total = np.bincount(bins, weights=p, minlength=r_max + 1)[1 : r_max + 1]
        azimuthal[sector_id] = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    scalars = scalar_features(rapsd, azimuthal)
    flat = log_power[triangle].astype(np.float32)
    return flat, scalars

def scalar_features(rapsd: np.ndarray, azimuthal: np.ndarray) -> np.ndarray:
    spectrum = rapsd.astype(np.float64)
    total = spectrum.sum()
    if total <= 0:
        return np.zeros(12, dtype=np.float32)
    normalized = spectrum / total
    radii = np.arange(1, len(normalized) + 1, dtype=np.float64)
    peak_average = float(normalized.max() / max(normalized.mean(), 1e-18))
    peak_radius = int(np.argmax(normalized[: len(normalized) // 2]))
    fit_start = min(max(peak_radius, len(normalized) // 2), len(normalized) - 10)
    if fit_start < len(normalized) - 5:
        slope, intercept = np.polyfit(
            np.log(radii[fit_start:]), np.log(normalized[fit_start:] + 1e-18), 1
        )
        alpha, beta = float(-slope), float(intercept)
    else:
        alpha, beta = 0.0, 0.0
    edges = np.linspace(1, len(normalized) + 1, 9).astype(int)
    energies = [float(normalized[edges[i] - 1 : edges[i + 1] - 1].sum()) for i in range(8)]
    sector_sums = azimuthal.sum(axis=1, keepdims=True)
    sector_norm = np.divide(azimuthal, sector_sums, out=np.zeros_like(azimuthal), where=sector_sums > 0)
    directional_variance = float(np.var(sector_norm, axis=0).mean())
    return np.asarray([peak_average, alpha, beta, *energies, directional_variance], dtype=np.float32)

def create_visual_model() -> tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained DINOv2 through torch.hub. Enable Kaggle internet access."
        ) from exc
    model.to(device).eval()
    return model, device

def visual_features(rgb: np.ndarray, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    image = Image.fromarray(rgb).resize((224, 224), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    vector = model(tensor)
    vector = F.normalize(vector, p=2, dim=-1)
    return vector[0].detach().cpu().numpy().astype(np.float32)

def image_feature(encoded: bytes, model: torch.nn.Module, device: torch.device, grids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(io.BytesIO(encoded)) as opened:
        opened.load()
        rgb = center_crop(np.asarray(opened.convert("RGB"), dtype=np.uint8))
    flat, scalars = spectral_features(rgb, grids)
    visual = visual_features(rgb, model, device)
    return flat, visual, scalars

def degraded_bytes(encoded: bytes, index: int) -> tuple[bytes, str]:
    with Image.open(io.BytesIO(encoded)) as opened:
        image = opened.convert("RGB")
    buffer = io.BytesIO()
    if index % 3 == 0:
        image.save(buffer, format="JPEG", quality=18, optimize=True)
        kind = "severe_jpeg"
    elif index % 3 == 1:
        width, height = image.size
        tiny = image.resize((max(8, width // 8), max(8, height // 8)), Image.Resampling.BILINEAR)
        tiny.resize((width, height), Image.Resampling.BICUBIC).save(buffer, format="JPEG", quality=55)
        kind = "severe_downscale"
    else:
        image.filter(ImageFilter.GaussianBlur(radius=8)).save(buffer, format="PNG")
        kind = "severe_blur"
    return buffer.getvalue(), kind

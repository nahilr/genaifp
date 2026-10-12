"""Shared constants for the standalone CommunityForensics Kaggle workflow."""
from __future__ import annotations

from pathlib import Path
import numpy as np

SHARDS = [
    (
        "OwensLab/CommunityForensics",
        "HFCF_Commercial_0.parquet",
        "https://huggingface.co/datasets/OwensLab/CommunityForensics/resolve/main/data/commercial/HFCF_Commercial_0.parquet",
    ),
    (
        "OwensLab/CommunityForensics",
        "HFCF_Commercial_1.parquet",
        "https://huggingface.co/datasets/OwensLab/CommunityForensics/resolve/main/data/commercial/HFCF_Commercial_1.parquet",
    ),
    (
        "OwensLab/CommunityForensics-Small",
        "HFCF_small_174.parquet",
        "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small/resolve/main/data/HFCF_small_174.parquet",
    ),
]

TARGET_MODELS = {
    "dalle2": "dalle2",
    "dalle3": "dalle3",
    "midjourneyv52": "midjourney_v5_2",
    "midjourneyv61": "midjourney_v6_1",
    "fireflyimage2": "firefly_image2",
    "fireflyimage3": "firefly_image3",
    "fluxdev": "flux_dev",
    "fluxschnell": "flux_schnell",
    "ideogramv1": "ideogram_v1",
    "ideogramv2": "ideogram_v2",
    "imagen3": "imagen3",
}
REAL_MODELS = {"landscapeshq", "coco", "ffhq", "vision"}
TARGET_LABELS = sorted([*TARGET_MODELS.values(), "real_camera"])
MAX_PER_CLASS = 500
MAX_UNKNOWN_PER_MODEL = 25
MAX_UNKNOWN_TOTAL = 200
IMAGE_EXPORT_PER_CLASS = 10
CROP_SIZE = 512
PCA_COMPONENTS = 128
RANGE_CHUNK_BYTES = 64 * 1024 * 1024
SRM_KERNEL = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float64)
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)

WORK = Path("/kaggle/working")
RAW_DIR = WORK / "communityforensics_raw"
OUTPUT_DIR = WORK / "genaifp-results"
IMAGE_DIR = OUTPUT_DIR / "extracted-images"
IMAGE_CANDIDATE_DIR = WORK / "communityforensics-image-candidates"
FEATURES_PATH = OUTPUT_DIR / "features_communityforensics_real.parquet"

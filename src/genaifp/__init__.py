"""GenAI Fingerprint (genaifp): Generative AI Engine Fingerprinting & Forensic Attribution

A production forensic attribution pipeline that identifies the specific generative model
(Midjourney, DALL·E, Imagen, Ideogram, Real Camera) responsible for synthesizing an image.
"""

from .preprocessor import (
    CROP_SIZE,
    SRM_KERNEL,
    Y_COEFFS,
    apply_srm_highpass,
    load_and_crop,
    preprocess_array,
    preprocess_batch,
    preprocess_image,
    rgb_to_luminance,
)
from .spectral_engine import (
    SpectralEngine,
    compute_2d_spectrum,
    compute_azimuthal_grid,
    compute_radial_grid,
    extract_azimuthal_profiles,
    extract_rapsd,
    flatten_upper_triangular,
)

try:
    from .visual_engine import (
        VisualEngine,
        extract_visual_features,
    )
except ImportError:
    VisualEngine = None
    extract_visual_features = None

try:
    from .feature_extractor import (
        FEATURE_NAMES_12,
        HYBRID_FEATURE_DIM,
        VISUAL_FEATURE_DIM,
        FeatureExtractor,
    )
except ImportError:
    FeatureExtractor = None
    FEATURE_NAMES_12 = []
    HYBRID_FEATURE_DIM = 524
    VISUAL_FEATURE_DIM = 384

try:
    from .classifier import (
        AttributionClassifier,
        train_attribution_model,
    )
except ImportError:
    AttributionClassifier = None
    train_attribution_model = None

try:
    from .robustness_bench import (
        PERTURBATIONS,
        RobustnessBenchmark,
        apply_downscale_upscale,
        apply_gaussian_blur,
        apply_jpeg,
        run_robustness_benchmark,
    )
except ImportError:
    RobustnessBenchmark = None
    run_robustness_benchmark = None
    apply_jpeg = None
    apply_gaussian_blur = None
    apply_downscale_upscale = None
    PERTURBATIONS = {}

__version__ = "0.1.0"

__all__ = [
    # Preprocessor
    "preprocess_image",
    "preprocess_array",
    "preprocess_batch",
    "load_and_crop",
    "rgb_to_luminance",
    "apply_srm_highpass",
    "SRM_KERNEL",
    "Y_COEFFS",
    "CROP_SIZE",
    # Spectral engine
    "SpectralEngine",
    "compute_2d_spectrum",
    "compute_radial_grid",
    "compute_azimuthal_grid",
    "extract_rapsd",
    "extract_azimuthal_profiles",
    "flatten_upper_triangular",
    # Feature extractor
    "FeatureExtractor",
    "FEATURE_NAMES_12",
    # Classifier
    "AttributionClassifier",
    "train_attribution_model",
    # Robustness
    "RobustnessBenchmark",
    "run_robustness_benchmark",
    "apply_jpeg",
    "apply_gaussian_blur",
    "apply_downscale_upscale",
    "PERTURBATIONS",
]

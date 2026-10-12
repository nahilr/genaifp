"""Module 5: Robustness Benchmark.

Stress-tests attribution against common perturbations:
- JPEG compression (QF 95, 85, 75, 65)
- Gaussian blur (σ=0.5, 1.0, 1.5)
- Downscale-upscale (0.75× then restore to 512×512)
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageFilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .classifier import AttributionClassifier
from .feature_extractor import FeatureExtractor
from .spectral_engine import SpectralEngine


def apply_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression to uint8 RGB image."""
    # image: (H, W, 3) uint8
    pil_img = Image.fromarray(image.astype(np.uint8), "RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")
    return np.asarray(compressed, dtype=np.uint8)


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to uint8 RGB image."""
    pil_img = Image.fromarray(image.astype(np.uint8), "RGB")
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred, dtype=np.uint8)


def apply_downscale_upscale(image: np.ndarray, scale: float = 0.75) -> np.ndarray:
    """Downscale by factor then restore to original size (bicubic)."""
    H, W = image.shape[:2]
    new_H, new_W = int(H * scale), int(W * scale)
    pil_img = Image.fromarray(image.astype(np.uint8), "RGB")
    downscaled = pil_img.resize((new_W, new_H), Image.BICUBIC)
    restored = downscaled.resize((W, H), Image.BICUBIC)
    return np.asarray(restored, dtype=np.uint8)


PERTURBATIONS = {
    "jpeg": {
        "func": apply_jpeg,
        "params": {"quality": [95, 85, 75, 65]},
        "label_fn": lambda q: f"JPEG QF={q}",
    },
    "blur": {
        "func": apply_gaussian_blur,
        "params": {"sigma": [0.5, 1.0, 1.5]},
        "label_fn": lambda s: f"Blur σ={s}",
    },
    "downscale": {
        "func": apply_downscale_upscale,
        "params": {"scale": [0.75]},
        "label_fn": lambda s: f"Downscale {s}×",
    },
}


class RobustnessBenchmark:
    """Run robustness evaluation on a trained classifier."""

    def __init__(
        self,
        classifier: AttributionClassifier,
        feature_extractor: FeatureExtractor,
        spectral_engine: SpectralEngine,
    ):
        self.clf = classifier
        self.feat_ext = feature_extractor
        self.spec_engine = spectral_engine

    def _extract_from_pil(self, pil_img: Image.Image) -> dict[str, np.ndarray]:
        """Extract features from a PIL image."""
        # Convert to uint8 array
        arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        # Full pipeline
        residual = self._preprocess_array(arr)
        spectral = self.spec_engine.process(residual)
        features = self.feat_ext.extract(spectral)
        return features

    def _preprocess_array(self, arr: np.ndarray) -> np.ndarray:
        """Preprocess uint8 RGB array (H, W, 3) -> residual (512, 512) float32."""
        from .preprocessor import preprocess_array

        return preprocess_array(arr, crop_size=self.spec_engine.crop_size, allow_pad=True)

    def evaluate_perturbation(
        self,
        clean_images: list[np.ndarray],
        clean_labels: list[str],
        perturbation_name: str,
        param_values: list,
        apply_fn: Callable,
        label_fn: Callable,
    ) -> list[dict]:
        """Evaluate classifier on perturbed images.

        Args:
            clean_images: list of (H, W, 3) uint8 clean images
            clean_labels: ground truth labels
            perturbation_name: name of perturbation type
            param_values: list of parameter values to test
            apply_fn: function(image, param) -> perturbed image
            label_fn: function(param) -> label string

        Returns:
            List of result dicts per parameter value
        """
        results = []

        for param in param_values:
            print(f"  Testing {perturbation_name} {label_fn(param)}...")

            # Apply perturbation
            perturbed = [apply_fn(img, param) for img in clean_images]

            # Extract features and predict
            preds = []
            confs = []
            n_expected = getattr(self.clf.model, "n_features_in_", 128)
            for img in perturbed:
                residual = self._preprocess_array(img)
                spectral = self.spec_engine.process(residual)
                features = self.feat_ext.extract(spectral)
                if n_expected == 524:
                    if not hasattr(self, "_vis_engine") or self._vis_engine is None:
                        from .visual_engine import VisualEngine

                        self._vis_engine = VisualEngine()
                    vis_feat = self._vis_engine.extract(img)
                    feature_vec = np.concatenate(
                        [vis_feat, features["pca_features"], features["scalar_12"]], axis=0
                    ).astype(np.float32)
                elif n_expected == 140:
                    feature_vec = np.concatenate(
                        [features["pca_features"], features["scalar_12"]], axis=0
                    ).astype(np.float32)
                else:
                    feature_vec = features["pca_features"]
                pred = self.clf.predict_single(feature_vec)
                preds.append(pred["predicted_engine"])
                confs.append(pred["confidence"])

            # Compute accuracy
            correct = sum(p == lbl for p, lbl in zip(preds, clean_labels, strict=False))
            acc = correct / len(clean_labels)
            mean_conf = np.mean(confs)

            # Per-class accuracy
            per_class = {}
            for cls in self.clf.class_names:
                cls_mask = np.array(clean_labels) == cls
                if cls_mask.any():
                    cls_correct = sum(
                        p == lbl
                        for p, lbl in zip(
                            np.array(preds)[cls_mask],
                            np.array(clean_labels)[cls_mask],
                            strict=False,
                        )
                    )
                    per_class[cls] = cls_correct / cls_mask.sum()

            results.append(
                {
                    "perturbation": perturbation_name,
                    "parameter": param,
                    "label": label_fn(param),
                    "accuracy": acc,
                    "mean_confidence": float(mean_conf),
                    "per_class_accuracy": per_class,
                    "predictions": preds,
                    "confidences": confs,
                }
            )

        return results

    def run_full_benchmark(
        self, clean_images: list[np.ndarray], clean_labels: list[str], output_dir: str | Path
    ) -> dict:
        """Run full robustness benchmark suite."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}

        for pert_name, config in PERTURBATIONS.items():
            print(f"\n=== {pert_name.upper()} ===")
            results = self.evaluate_perturbation(
                clean_images,
                clean_labels,
                pert_name,
                config["params"][list(config["params"].keys())[0]],
                config["func"],
                config["label_fn"],
            )
            all_results[pert_name] = results

        # Save results
        with open(output_dir / "robustness_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Generate plots
        self._plot_results(all_results, output_dir)

        return all_results

    def _plot_results(self, results: dict, output_dir: Path) -> None:
        """Generate robustness degradation curves."""
        sns.set_style("whitegrid")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, (pert_name, res_list) in zip(axes, results.items(), strict=False):
            labels = [r["label"] for r in res_list]
            accs = [r["accuracy"] for r in res_list]
            confs = [r["mean_confidence"] for r in res_list]

            ax.plot(range(len(labels)), accs, "o-", label="Accuracy", color="steelblue")
            ax.plot(range(len(labels)), confs, "s--", label="Mean Confidence", color="orange")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=15)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title(pert_name.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "robustness_curves.png", dpi=150)
        plt.close()

        # Per-class heatmap
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
        if len(results) == 1:
            axes = [axes]

        for ax, (pert_name, res_list) in zip(axes, results.items(), strict=False):
            classes = self.clf.class_names
            matrix = np.zeros((len(classes), len(res_list)))

            for j, r in enumerate(res_list):
                for i, cls in enumerate(classes):
                    matrix[i, j] = r["per_class_accuracy"].get(cls, 0)

            sns.heatmap(
                matrix,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                xticklabels=[r["label"] for r in res_list],
                yticklabels=classes,
                ax=ax,
                vmin=0,
                vmax=1,
            )
            ax.set_title(f"{pert_name.capitalize()} - Per-Class Accuracy")

        plt.tight_layout()
        plt.savefig(output_dir / "robustness_per_class.png", dpi=150)
        plt.close()


def run_robustness_benchmark(
    model_path: str | Path,
    feat_extractor_path: str | Path,
    test_images: list[np.ndarray],
    test_labels: list[str],
    output_dir: str | Path,
    crop_size: int = 512,
) -> dict:
    """Convenience function to run full benchmark.

    Args:
        model_path: path to saved classifier (.json/.pkl)
        feat_extractor_path: path to saved feature extractor (.pkl)
        test_images: list of clean test images (uint8 RGB)
        test_labels: ground truth labels
        output_dir: directory for results
        crop_size: image crop size
    """
    # Load components
    clf = AttributionClassifier.load(model_path)
    feat_ext = FeatureExtractor.load(feat_extractor_path)
    spec_engine = SpectralEngine(crop_size=crop_size)

    bench = RobustnessBenchmark(clf, feat_ext, spec_engine)
    return bench.run_full_benchmark(test_images, test_labels, output_dir)


if __name__ == "__main__":
    # Quick self-test

    # Synthetic test image
    test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Test perturbations
    print("Testing perturbations...")
    jpeg_img = apply_jpeg(test_img, 85)
    blur_img = apply_gaussian_blur(test_img, 1.0)
    down_img = apply_downscale_upscale(test_img, 0.75)

    print(f"JPEG 85: {jpeg_img.shape}, range=[{jpeg_img.min()}, {jpeg_img.max()}]")
    print(f"Blur 1.0: {blur_img.shape}")
    print(f"Down 0.75: {down_img.shape}")

    print("Perturbation functions OK")

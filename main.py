#!/usr/bin/env python3
"""GenAI Fingerprint (genaifp) - CLI Tool Entrypoint.

Usage:
    python main.py <command> [options]
    python main.py <path_to_image>
    python main.py --image_path <path_to_image>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd

from genaifp.classifier import AttributionClassifier, train_attribution_model
from genaifp.data_pipeline import GROUPING_METHOD
from genaifp.feature_extractor import FeatureExtractor
from genaifp.preprocessor import CROP_SIZE, preprocess_array, preprocess_image
from genaifp.robustness_bench import apply_downscale_upscale, apply_gaussian_blur, apply_jpeg
from genaifp.spectral_engine import SpectralEngine

# =====================================================================
# 1. PREDICT CLI
# =====================================================================


def parse_predict_args(args=None):
    parser = argparse.ArgumentParser(description="Single-image forensic attribution")
    parser.add_argument(
        "--model_path",
        default="models/attribution_hybrid",
        help="Path to evaluated model (without extension; defaults to models/attribution_hybrid)",
    )
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE, help="Crop size")
    parser.add_argument("--output_json", help="Optional output JSON file")
    return parser.parse_args(args)


def predict_main(args=None):
    parsed = parse_predict_args(args)

    model_path = Path(parsed.model_path)
    image_path = Path(parsed.image_path)

    manifest_path = model_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        print(
            f"ERROR: model has no evaluated real-data manifest: {manifest_path}. "
            "Train a CommunityForensics model before prediction.",
            file=sys.stderr,
        )
        return 1
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: invalid model manifest: {exc}", file=sys.stderr)
        return 1
    if not (
        manifest.get("dataset_kind") == "communityforensics"
        and manifest.get("evaluation_status") == "research_holdout_complete"
        and manifest.get("evaluated_on_heldout_groups") is True
        and manifest.get("grouping_method") == GROUPING_METHOD
        and manifest.get("probabilities_calibrated") is True
        and manifest.get("model_type") == "hybrid_524d"
    ):
        print("ERROR: model is a demo or has not completed real-data validation.", file=sys.stderr)
        return 1
    model_crop_size = int(manifest.get("preprocessing", {}).get("crop_size", 0))
    if parsed.crop_size != model_crop_size:
        print(
            f"ERROR: model requires crop_size={model_crop_size}; received {parsed.crop_size}.",
            file=sys.stderr,
        )
        return 1
    feat_path = model_path.parent / str(manifest.get("feature_extractor", ""))

    for p, name in [
        (model_path.with_suffix(".json"), "model"),
        (model_path.with_suffix(".pkl"), "model metadata"),
        (feat_path, "feature extractor"),
        (image_path, "input image"),
    ]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}", file=sys.stderr)
            return 1

    # Load components
    print(f"Loading model from {model_path}...")
    clf = AttributionClassifier.load(model_path)

    print(f"Loading feature extractor from {feat_path}...")
    feat_ext = FeatureExtractor.load(feat_path)
    if (
        not clf.probabilities_calibrated
        or clf.ood_detector is None
        or getattr(clf.model, "n_features_in_", None) != 524
        or list(clf.class_names) != list(manifest.get("classes", []))
        or getattr(feat_ext.pca, "n_components_", None) != 128
    ):
        print("ERROR: model files do not match the calibrated manifest.", file=sys.stderr)
        return 1

    print("Initializing spectral engine...")
    spec_engine = SpectralEngine(crop_size=parsed.crop_size)

    # Process image
    print(f"Processing {image_path}...")
    residual = preprocess_image(image_path, crop_size=parsed.crop_size, allow_pad=True)

    # Spectral features
    spectral = spec_engine.process(residual)
    features = feat_ext.extract(spectral)

    n_expected = getattr(clf.model, "n_features_in_", 128)
    if n_expected != 524:
        print("ERROR: model feature dimension does not match its hybrid manifest.", file=sys.stderr)
        return 1

    # Check feature dimension
    if n_expected == 524:
        from genaifp.visual_engine import VisualEngine

        print("Extracting DINOv2 visual features (384 dims)...")
        vis_engine = VisualEngine(crop_size=parsed.crop_size)
        vis_feat = vis_engine.extract(image_path)
        feature_vec = np.concatenate(
            [vis_feat, features["pca_features"], features["scalar_12"]], axis=0
        ).astype(np.float32)
    elif n_expected == 140:
        feature_vec = np.concatenate(
            [features["pca_features"], features["scalar_12"]], axis=0
        ).astype(np.float32)
    else:
        feature_vec = features["pca_features"]

    # Predict
    result = clf.predict_single(feature_vec)

    # Add spectral metrics
    rapsd = spectral["rapsd"]
    azimuthal = spectral["azimuthal"]

    Pn = rapsd / (rapsd.sum() + 1e-12)
    r = np.arange(1, len(Pn) + 1)
    peak_r = int(np.argmax(Pn[: len(Pn) // 2]))
    fit_start = max(peak_r, len(Pn) // 2)
    if fit_start < len(Pn) - 5:
        mask = r >= r[fit_start]
        log_r = np.log(r[mask])
        log_P = np.log(Pn[mask] + 1e-18)
        slope, intercept = np.polyfit(log_r, log_P, 1)
        alpha = float(-slope)
        beta = float(intercept)
    else:
        alpha, beta = 0.0, 0.0

    pasr = float(Pn.max() / (Pn.mean() + 1e-12))

    A = azimuthal.astype(np.float64)
    A_sum = A.sum(axis=1, keepdims=True)
    A_norm = np.divide(A, A_sum, out=np.zeros_like(A), where=A_sum > 0)
    dir_var = float(np.mean(np.var(A_norm, axis=0)))

    result["spectral_metrics"] = {
        "pasr": pasr,
        "spectral_slope_alpha": alpha,
        "spectral_intercept_beta": beta,
        "directional_variance": dir_var,
    }
    result["architecture"] = "hybrid_524d" if n_expected == 524 else f"spectral_{n_expected}d"
    result["model_provenance"] = {
        "dataset_id": manifest.get("dataset_id"),
        "evaluation_status": manifest.get("evaluation_status"),
        "held_out_test_samples": manifest.get("sample_counts", {}).get("test_known"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "grouping_method": manifest.get("grouping_method"),
        "grouping_limitations": manifest.get("grouping_limitations", []),
    }
    result["notice"] = "Experimental research output. This result is not forensic evidence."

    pred = result["predicted_engine"]
    conf = result["confidence"]
    if n_expected == 524:
        result["forensic_summary"] = (
            f"Image attributed to '{pred}' ({conf:.1%} confidence). "
            f"Analyzed using fused DINOv2 visual foundation embeddings + spectral residual lattice (PASR: {pasr:.2f}, Slope α: {alpha:.2f})."
        )
    else:
        result["forensic_summary"] = (
            f"Image attributed to '{pred}' ({conf:.1%} confidence) using spectral Fourier residual lattice."
        )

    output = json.dumps(result, indent=2)

    if parsed.output_json:
        Path(parsed.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(parsed.output_json, "w") as f:
            f.write(output)
        print(f"Result saved to {parsed.output_json}")
    else:
        print(output)

    return 0


# =====================================================================
# 2. TRAIN CLI
# =====================================================================


def parse_train_args(args=None):
    parser = argparse.ArgumentParser(description="Train attribution classifier")
    parser.add_argument(
        "--input", default="data/processed/features_hybrid.parquet", help="Input features parquet"
    )
    parser.add_argument(
        "--model_out",
        default="models/attribution_hybrid",
        help="Output model path (without extension)",
    )
    parser.add_argument("--report_dir", default="reports", help="Directory for evaluation reports")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--cv_folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=500, help="XGBoost n_estimators")
    parser.add_argument("--max_depth", type=int, default=6, help="XGBoost max_depth")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="XGBoost learning_rate")
    parser.add_argument(
        "--feature_col",
        default=None,
        help="Feature column name (auto-detects hybrid_features or pca_features)",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Explicitly train a synthetic demonstration model (never eligible for prediction/web use)",
    )
    return parser.parse_args(args)


def train_main(args=None):
    parsed = parse_train_args(args)

    if not parsed.demo_only:
        print(
            "ERROR: full training, calibration, and evaluation run in Kaggle. "
            "Use the Kaggle notebook, then run `download_outputs.py` and `install_outputs.py`. "
            "Local training is available only for synthetic demos with --demo-only.",
            file=sys.stderr,
        )
        return 1

    input_path = Path(parsed.input)
    model_out = Path(parsed.model_out)
    report_dir = Path(parsed.report_dir)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {input_path}...")
    try:
        summary_df = pd.read_parquet(input_path, columns=["label"])
    except Exception as exc:
        print(f"ERROR: could not read feature metadata: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(summary_df)} samples")
    print(f"Classes: {summary_df['label'].unique()}")
    print(f"Class distribution:\n{summary_df['label'].value_counts()}")

    print("\n=== Training ===")
    effective_model_out = model_out
    is_demo_model = False
    try:
        import pyarrow.parquet as pq

        kinds = set()
        if "dataset_kind" in pq.read_schema(input_path).names:
            kinds = set(pd.read_parquet(input_path, columns=["dataset_kind"])["dataset_kind"].dropna())
        if kinds == {"communityforensics"}:
            from genaifp.research import train_research_model

            results = train_research_model(
                input_path,
                model_out,
                report_dir,
                random_state=parsed.random_state,
                n_estimators=parsed.n_estimators,
                max_depth=parsed.max_depth,
                learning_rate=parsed.learning_rate,
                cv_folds=parsed.cv_folds,
            )
        elif parsed.demo_only:
            is_demo_model = True
            demo_model_out = model_out
            if demo_model_out == Path("models/attribution_hybrid"):
                demo_model_out = Path("models/demo_attribution")
            effective_model_out = demo_model_out
            results = train_attribution_model(
                features_path=input_path,
                model_out=demo_model_out,
                report_dir=report_dir,
                test_size=parsed.test_size,
                random_state=parsed.random_state,
                cv_folds=parsed.cv_folds,
                n_estimators=parsed.n_estimators,
                max_depth=parsed.max_depth,
                learning_rate=parsed.learning_rate,
                feature_col=parsed.feature_col,
            )
            demo_manifest = {
                "format_version": 1,
                "dataset_kind": "demo",
                "evaluation_status": "synthetic_demo_only",
                "evaluated_on_heldout_groups": False,
            }
            demo_model_out.with_suffix(".manifest.json").write_text(
                json.dumps(demo_manifest, indent=2), encoding="utf-8"
            )
        else:
            raise ValueError(
                "Input is not a CommunityForensics feature file. Use --demo-only only for a synthetic demonstration."
            )
    except Exception as exc:
        print(f"ERROR: training failed: {exc}", file=sys.stderr)
        return 1

    test_res = results["test"]
    cv_res = results["cv"]

    print("\n=== FINAL RESULTS ===")
    print(f"Test Accuracy:  {test_res['accuracy']:.4f}")
    print(f"Test Macro F1:  {test_res['macro_f1']:.4f}")
    cv_mean = cv_res.get("macro_f1_mean", cv_res.get("cv_f1_macro_mean", 0.0))
    cv_std = cv_res.get("macro_f1_std", cv_res.get("cv_f1_macro_std", 0.0))
    print(f"CV Macro F1:    {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Weighted F1:    {test_res['weighted_f1']:.4f}")

    print("\nPer-class F1:")
    for cls, metrics in test_res["per_class"].items():
        print(
            f"  {cls:18s}: F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  (n={metrics['support']})"
        )

    macro_f1 = test_res["macro_f1"]
    if is_demo_model:
        print(f"Demo model only: synthetic score {macro_f1:.2%}; not a generator attribution metric.")
    else:
        print("Evaluation uses held-out prompt groups. Treat all scores as research results.")

    summary = {
        "dataset_kind": "demo" if is_demo_model else "communityforensics",
        "test_accuracy": test_res["accuracy"],
        "test_macro_f1": test_res["macro_f1"],
        "test_weighted_f1": test_res["weighted_f1"],
        "cv_macro_f1_mean": cv_mean,
        "cv_macro_f1_std": cv_std,
        "per_class_f1": {cls: m["f1"] for cls, m in test_res["per_class"].items()},
        "research_only": True,
    }

    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {report_dir / 'summary.json'}")
    print(f"Model saved to {effective_model_out}.json / .pkl")

    return 0


# =====================================================================
# 3. BUILD DATASET CLI
# =====================================================================

MODEL_TO_CLASS = {
    "MidjourneyV6_1": "midjourney_v6",
    "MidjourneyV5_2": "midjourney_v5_2",
    "Imagen3": "imagen3",
    "IdeogramV2": "ideogram_v2",
    "Dalle2": "dalle2",
    "Dalle3": "dalle3",
    "LandscapesHQ": "real_camera",
    "COCO": "real_camera",
    "FFHQ": "real_camera",
    "VISION": "real_camera",
}

TARGET_CLASSES = [
    "midjourney_v6",
    "dalle3",
    "imagen3",
    "ideogram_v2",
    "dalle2",
    "real_camera",
]


def build_demo_dataset(
    output_path: Path, crop_size: int = 512, n_pca: int = 128, n_samples_per_class: int = 100
):
    from PIL import Image

    from genaifp.visual_engine import VisualEngine

    samples_dir = Path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    spec_engine = SpectralEngine(crop_size=crop_size)
    feat_ext = FeatureExtractor(n_pca_components=n_pca)
    print("Initializing VisualEngine (DINOv2-small)...")
    vis_engine = VisualEngine(crop_size=crop_size)

    rng = np.random.default_rng(42)
    all_flat = []
    all_rapsd = []
    all_azimuthal = []
    all_images = []
    metadata = []

    print(
        f"Generating demo dataset with {len(TARGET_CLASSES)} classes, {n_samples_per_class} samples each..."
    )

    for cls_name in TARGET_CLASSES:
        for idx in range(n_samples_per_class):
            x_grad = np.linspace(0, 1, crop_size, dtype=np.float32)
            y_grad = np.linspace(0, 1, crop_size, dtype=np.float32)
            xx, yy = np.meshgrid(x_grad, y_grad)
            smooth = (
                0.5 + 0.3 * np.sin(2.5 * xx + idx * 0.1) * np.cos(2.5 * yy + idx * 0.1)
            ) * 180 + 30
            noise = rng.normal(0, 6, (crop_size, crop_size))

            if cls_name == "real_camera":
                r_ch = np.clip(smooth + noise + 4, 0, 255)
                g_ch = np.clip(smooth + noise, 0, 255)
                b_ch = np.clip(smooth + noise - 4, 0, 255)
                g_ch[::2, 1::2] = np.clip(g_ch[::2, 1::2] + 2, 0, 255)
                g_ch[1::2, ::2] = np.clip(g_ch[1::2, ::2] + 2, 0, 255)
                base = np.stack([r_ch, g_ch, b_ch], axis=-1).astype(np.uint8)
            else:
                r_ch = np.clip(smooth + noise, 0, 255)
                g_ch = np.clip(smooth + noise, 0, 255)
                b_ch = np.clip(smooth + noise, 0, 255)
                base = np.stack([r_ch, g_ch, b_ch], axis=-1).astype(np.uint8)

                if cls_name == "midjourney_v6":
                    base[::8, ::8] = np.clip(base[::8, ::8].astype(int) + 38, 0, 255)
                    base[1::8, 1::8] = np.clip(base[1::8, 1::8].astype(int) - 38, 0, 255)
                    base[:, :, 0] = np.clip(base[:, :, 0].astype(int) + 14, 0, 255)
                elif cls_name == "dalle3":
                    base[::16, :] = np.clip(base[::16, :].astype(int) + 38, 0, 255)
                    base[:, ::16] = np.clip(base[:, ::16].astype(int) - 38, 0, 255)
                    base[:, :, 1] = np.clip(base[:, :, 1].astype(int) + 12, 0, 255)
                elif cls_name == "dalle2":
                    base[::8, :] = np.clip(base[::8, :].astype(int) + 30, 0, 255)
                    base[:, ::8] = np.clip(base[:, ::8].astype(int) - 30, 0, 255)
                    base[::16, ::16] = np.clip(base[::16, ::16].astype(int) + 25, 0, 255)
                elif cls_name == "imagen3":
                    base[::4, :] = np.clip(base[::4, :].astype(int) + 32, 0, 255)
                    base[:, ::4] = np.clip(base[:, ::4].astype(int) - 32, 0, 255)
                elif cls_name == "ideogram_v2":
                    base[::8, :] = np.clip(base[::8, :].astype(int) + 36, 0, 255)
                    base[:, :, 2] = np.clip(base[:, :, 2].astype(int) + 14, 0, 255)

            augmented = base
            aug_tag = "clean"
            if idx % 2 == 1:
                aug_type = idx % 4
                if aug_type == 0:
                    qf = 65 + (idx % 3) * 10
                    augmented = apply_jpeg(base, qf)
                    aug_tag = f"jpeg_qf{qf}"
                elif aug_type == 1:
                    augmented = apply_downscale_upscale(base, 0.75)
                    aug_tag = "downscale_0.75"
                elif aug_type == 2:
                    augmented = apply_gaussian_blur(base, 0.8)
                    aug_tag = "blur_0.8"
                elif aug_type == 3:
                    augmented = apply_jpeg(base, 80)
                    aug_tag = "jpeg_qf80"

            if idx == 0:
                sample_img_path = samples_dir / f"{cls_name}_sample.png"
                Image.fromarray(base).save(sample_img_path)

            all_images.append(augmented)
            residual = preprocess_array(augmented, crop_size=crop_size)
            spectral = spec_engine.process(residual)

            all_flat.append(spectral["flat_spectrum"])
            all_rapsd.append(spectral["rapsd"])
            all_azimuthal.append(spectral["azimuthal"])

            metadata.append(
                {
                    "label": cls_name,
                    "model_name": cls_name,
                    "image_name": f"{cls_name}_{idx:04d}.png",
                    "source_shard": f"demo_{aug_tag}",
                }
            )

    print(f"Extracting DINOv2 visual features for {len(all_images)} images...")
    vis_feats = vis_engine.extract_batch(all_images)

    all_flat_arr = np.stack(all_flat)
    print(f"Fitting spectral PCA on {len(all_flat_arr)} samples...")
    feat_ext.fit_pca(all_flat_arr)
    pca_feats = feat_ext.transform_pca(all_flat_arr)

    print("Computing physical spectral metrics...")
    scalars_12 = np.zeros((len(all_images), 12), dtype=np.float32)
    for i in range(len(all_images)):
        scalars_12[i] = feat_ext.extract_12_scalar(all_rapsd[i], all_azimuthal[i])

    print("Fusing into 524-dim hybrid feature vectors...")
    hybrid_feats = np.concatenate([vis_feats, pca_feats, scalars_12], axis=1).astype(np.float32)

    feat_ext_path = output_path.parent / "feature_extractor.pkl"
    feat_ext.save(feat_ext_path)
    models_feat_path = Path("models") / "feature_extractor.pkl"
    models_feat_path.parent.mkdir(parents=True, exist_ok=True)
    feat_ext.save(models_feat_path)

    results = []
    for i, meta in enumerate(metadata):
        results.append(
            {
                "hybrid_features": hybrid_feats[i],
                "pca_features": pca_feats[i],
                "visual_features": vis_feats[i],
                "scalar_12": scalars_12[i],
                "label": meta["label"],
                "model_name": meta["model_name"],
                "image_name": meta["image_name"],
                "source_shard": meta["source_shard"],
                "dataset_kind": "demo",
            }
        )

    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)

    hybrid_path = output_path.parent / "features_hybrid.parquet"
    df.to_parquet(hybrid_path, index=False)

    print(
        f"Hybrid features saved to {output_path} and {hybrid_path} ({len(df)} samples, {hybrid_feats.shape[1]} dims)"
    )
    print(f"Reference test images saved to {samples_dir}/")
    return results


def parse_build_args(args=None):
    parser = argparse.ArgumentParser(description="Build feature dataset from shards")
    parser.add_argument("--data_dir", default="data/raw", help="Directory with downloaded shards")
    parser.add_argument(
        "--output",
        default="data/processed/features_communityforensics_real.parquet",
        help="Output parquet path",
    )
    parser.add_argument("--crop_size", type=int, default=512, help="Crop size")
    parser.add_argument("--n_pca", type=int, default=128, help="PCA components")
    parser.add_argument(
        "--demo", action="store_true", help="Generate synthetic calibrated demo dataset"
    )
    parser.add_argument(
        "--demo_samples_per_class", type=int, default=100, help="Samples per class for demo mode"
    )
    parser.add_argument("--max_per_class", type=int, default=250)
    parser.add_argument("--max_unknown_per_model", type=int, default=50)
    parser.add_argument("--max_unknown_total", type=int, default=1000)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Input rows processed at once (keep small to bound RAM while decoding images)",
    )
    parser.add_argument(
        "--drop_image_bytes",
        action="store_true",
        help="Do not retain original image bytes in the feature Parquet output",
    )
    parser.add_argument("--no_degraded", action="store_true", help="Skip severe OOD degradation examples")
    return parser.parse_args(args)


def build_main(args=None):
    parsed = parse_build_args(args)
    output_path = Path(parsed.output)

    if parsed.demo:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print("Running in DEMO mode: generating calibrated demonstration dataset...")
        build_demo_dataset(
            output_path, parsed.crop_size, parsed.n_pca, parsed.demo_samples_per_class
        )
        return 0
    print(
        "ERROR: real-data feature extraction runs in Kaggle. "
        "Build and run the notebook under kaggle/; local feature generation is disabled.",
        file=sys.stderr,
    )
    return 1


# =====================================================================
# 4. DOWNLOAD SHARDS & CURATED
# =====================================================================


def download_main(args=None):
    print(
        "ERROR: source shard downloads are part of the Kaggle training workflow. "
        "For local use, run `python kaggle/download_outputs.py` and then `python kaggle/install_outputs.py`.",
        file=sys.stderr,
    )
    return 1


def download_curated_main(args=None):
    return download_main(args)


# =====================================================================
# 5. MAIN DISPATCHER & HELP
# =====================================================================


def print_help():
    help_text = """GenAI Fingerprint (genaifp): Generative AI Engine Attribution

Usage:
    python main.py <command> [options]
    python main.py <path_to_image>
    python main.py --image_path <path_to_image>

Available Commands:
    predict          Identify which AI engine synthesized an image
                     Options: --image_path <path> [--model_path <path>] [--crop_size <n>]
                              [--output_json <path>]

    web              Start the local web app (install fastapi, uvicorn, python-multipart)
                     Options: [--model <path>] [--host <addr>] [--port <n>]

    build            Build a synthetic demo dataset; real-data feature extraction runs in Kaggle
                     Aliases: build_dataset
                     Options: --demo [--demo_samples_per_class <n>] [--output <parquet>]
                              [--crop_size <n>] [--n_pca <n>] [--batch_size <n>]
                              [--drop_image_bytes] [--no_degraded]

    train            Train a synthetic demo model only; real-data training runs in Kaggle
                     Aliases: training
                     Options: --input <parquet> --demo-only [--model_out <path>]
                              [--report_dir <dir>] [--feature_col <name>] [--test_size <f>]
                              [--cv_folds <n>] [--random_state <n>] [--n_estimators <n>]
                              [--max_depth <n>] [--learning_rate <f>]

    download         Use kaggle/download_outputs.py, then kaggle/install_outputs.py
                     Aliases: download_curated

    download_shards  Source data downloads are handled by the Kaggle notebook
                     Aliases: download_shard

    help             Print this summary (also -h or --help)

Examples:
    python main.py <path_to_image>
    python main.py --image_path <path_to_image>
    python main.py predict --image_path <path_to_image>
    python main.py train --input data/processed/features_hybrid.parquet --model_out models/demo_attribution --demo-only
    python main.py build --demo
    python kaggle/download_outputs.py
    python kaggle/install_outputs.py
"""
    print(help_text)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help", "help"):
        print_help()
        return 0

    command = argv[0].lower()
    sub_args = argv[1:]

    # Direct image shortcut: python main.py --image_path <path> or python main.py <image.png>
    if (
        command == "--image_path"
        or command.startswith("--image")
        or (Path(command).exists() and Path(command).is_file())
    ):
        if Path(command).exists() and Path(command).is_file():
            return predict_main(["--image_path", command] + sub_args)
        return predict_main(argv)

    if command == "predict":
        return predict_main(sub_args)
    elif command in ("train", "training"):
        return train_main(sub_args)
    elif command in ("build", "build_dataset"):
        return build_main(sub_args)
    elif command in ("download", "download_curated"):
        return download_curated_main(sub_args)
    elif command in ("download_shards", "download_shard"):
        return download_main(sub_args)
    elif command == "web":
        from genaifp.web import main as web_main

        return web_main(sub_args)
    else:
        print(f"Unknown command: '{command}'\n", file=sys.stderr)
        print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

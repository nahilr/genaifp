"""Orchestrate the standalone CommunityForensics Kaggle run."""
from __future__ import annotations

import json
import hashlib
import platform
import warnings
from datetime import UTC, datetime
import shutil
from pathlib import Path

import numpy as np
import torch

from config import CROP_SIZE, FEATURES_PATH, IMAGE_CANDIDATE_DIR, IMAGE_EXPORT_PER_CLASS, OUTPUT_DIR, RAW_DIR, TARGET_LABELS, WORK
from dataset_build import build_feature_rows
from features import create_visual_model
from training import calibrate_and_evaluate


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before running this script")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    warnings.filterwarnings("ignore", category=UserWarning)
    for artifact in (
        WORK / "genaifp-model-bundle.zip",
        WORK / "genaifp-test-images.zip",
        WORK / "genaifp-reports.zip",
    ):
        artifact.unlink(missing_ok=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    shutil.rmtree(IMAGE_CANDIDATE_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading DINOv2 pretrained visual model")
    vision_model, device = create_visual_model()
    rows, summary = build_feature_rows(vision_model, device)
    del vision_model
    torch.cuda.empty_cache()
    (OUTPUT_DIR / "import_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Imported rows:", len(rows), "summary:", json.dumps(summary))
    evaluation = calibrate_and_evaluate(rows, OUTPUT_DIR)
    if evaluation["manual_test_images"]["total"] != len(TARGET_LABELS) * 10:
        raise RuntimeError("Expected exactly 10 held-out test images for every target class")
    dataset_digest = hashlib.sha256()
    with FEATURES_PATH.open("rb") as feature_file:
        for chunk in iter(lambda: feature_file.read(1024 * 1024), b""):
            dataset_digest.update(chunk)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "created_at": datetime.now(UTC).isoformat(),
        "entrypoint": "main.py (standalone; no local project package installed)",
    }
    (OUTPUT_DIR / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    model_manifest = {
        "format_version": 1,
        "model_type": "hybrid_524d",
        "dataset_id": summary["dataset_id"],
        "dataset_kind": "communityforensics",
        "dataset_sha256": dataset_digest.hexdigest(),
        "evaluation_status": "research_holdout_complete",
        "evaluation_type": "internal_prompt_grouped_holdout",
        "evaluation_note": (
            "Commercial-source rows are marked test in the source dataset. These partitions are an "
            "internal grouped evaluation and are not the dataset's official benchmark score."
        ),
        "evaluated_on_heldout_groups": True,
        "grouping_method": "normalized_prompt_hash; exact-image-hash fallback when prompt is absent",
        "grouping_limitations": [
            "The dataset does not expose generation-batch IDs or unique source-photo IDs; prompt-disjoint groups cannot guarantee source-image-disjoint partitions."
        ],
        "classes": TARGET_LABELS,
        "feature_extractor": "feature_extractor.pkl",
        "feature_dimensions": {"visual": 384, "spectral_pca": 128, "scalar": 12},
        "visual_model": "facebookresearch/dinov2_vits14",
        "preprocessing": {"crop_size": CROP_SIZE, "color": "RGB", "crop_policy": "center-reflect-pad"},
        "source_shards": summary["source_shards"],
        "sample_counts": evaluation["split_counts"],
        "probabilities_calibrated": True,
        "temperature": evaluation["calibration"]["temperature"],
        "ood_threshold": evaluation["test"]["ood_threshold"],
        "ood_calibration": {
            "validation_false_reject_rate": evaluation["calibration"]["validation_false_reject_rate"],
            "validation_ood_recall": evaluation["calibration"]["validation_ood_recall"],
            "validation_ood_auroc": evaluation["calibration"]["validation_ood_auroc"],
        },
        "created_at": environment["created_at"],
        "kaggle_kernel": {"handle": "nahilr/genaifp"},
        "sampling": {
            "max_per_target_class": summary["sampling"]["max_per_target_class"],
            "max_unsupported_per_generator": summary["sampling"]["max_unsupported_per_generator"],
            "max_unsupported_total": summary["sampling"]["max_unsupported_total"],
            "unsupported_generator_examples_found": summary["unknown_total"],
            "degraded_examples_included": True,
            "original_image_bytes_retained_in_features": False,
        },
        "source_provenance": {
            "source_shard": summary["source_shards"],
            "source_split": sorted({str(row["source_split"]) for row in rows if row.get("source_split")}),
            "source_label": sorted({str(row["source_label"]) for row in rows if row.get("source_label")}),
            "subset": sorted({str(row["subset"]) for row in rows if row.get("subset")}),
            "model_name": sorted({str(row["model_name"]) for row in rows if row.get("model_name")}),
            "real_source": sorted({str(row["real_source"]) for row in rows if row.get("real_source")}),
            "original_source_split_counts": summary["original_source_split_counts"],
        },
        "original_image_bytes_retained_in_features": False,
    }
    manifest_path = OUTPUT_DIR / "attribution_hybrid.manifest.json"
    manifest_path.write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    print("Evaluation:", json.dumps(evaluation, indent=2))
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    shutil.rmtree(IMAGE_CANDIDATE_DIR, ignore_errors=True)
    FEATURES_PATH.unlink(missing_ok=True)
    print(f"Training artifacts are ready in: {OUTPUT_DIR}")
    print(f"Held-out sample images are ready in: {OUTPUT_DIR / 'extracted-images'}")


if __name__ == "__main__":
    main()

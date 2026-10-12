"""Leakage-resistant research training for provenance-rich image features."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .classifier import AttributionClassifier
from .data_pipeline import DATASET_ID, GROUPING_METHOD, TARGET_LABELS
from .feature_extractor import FeatureExtractor


def _hybrid_features(rows: pd.DataFrame, extractor: FeatureExtractor) -> np.ndarray:
    raw = np.stack(rows["flat_spectrum"].map(lambda value: np.asarray(value, dtype=np.float32)))
    pca = extractor.transform_pca(raw)
    visual = np.stack(rows["visual_features"].map(lambda value: np.asarray(value, dtype=np.float32)))
    scalars = np.stack(rows["scalar_12"].map(lambda value: np.asarray(value, dtype=np.float32)))
    return np.concatenate([visual, pca, scalars], axis=1).astype(np.float32)


def _select_group_splits(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose group-disjoint 70/15/15 train/calibration/test partitions."""
    strata = np.where(frame["is_ood"].to_numpy(dtype=bool), "ood", frame["label"].astype(str))
    groups = frame["group_id"].astype(str).to_numpy()
    target_classes = set(TARGET_LABELS)
    group_counts = pd.DataFrame({"stratum": strata, "group": groups}).groupby("stratum")["group"].nunique()
    outer_folds = min(5, int(group_counts.min()))
    if outer_folds < 3:
        raise ValueError("Need at least three independent groups per target/OOD stratum")
    outer = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    for development, test in outer.split(np.zeros(len(frame)), strata, groups):
        test_rows = frame.iloc[test]
        if not target_classes.issubset(set(test_rows.loc[~test_rows["is_ood"], "label"])):
            continue
        if not test_rows["is_ood"].any():
            continue
        inner_frame = frame.iloc[development].reset_index(drop=False).rename(columns={"index": "source_index"})
        inner_strata = strata[development]
        inner_groups = groups[development]
        inner_group_counts = (
            pd.DataFrame({"stratum": inner_strata, "group": inner_groups})
            .groupby("stratum")["group"]
            .nunique()
        )
        inner_folds = min(5, int(inner_group_counts.min()))
        if inner_folds < 2:
            continue
        inner = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed + 1)
        for train_local, calibration_local in inner.split(
            np.zeros(len(inner_frame)), inner_strata, inner_groups
        ):
            train_rows = inner_frame.iloc[train_local]
            calibration_rows = inner_frame.iloc[calibration_local]
            train_known = train_rows.loc[~train_rows["is_ood"], "label"]
            calibration_known = calibration_rows.loc[~calibration_rows["is_ood"], "label"]
            if (
                target_classes.issubset(set(train_known))
                and target_classes.issubset(set(calibration_known))
                and calibration_rows["is_ood"].any()
            ):
                train = inner_frame.iloc[train_local]["source_index"].to_numpy(dtype=int)
                calibration = inner_frame.iloc[calibration_local]["source_index"].to_numpy(dtype=int)
                return train, calibration, test
    raise ValueError(
        "Could not create group-disjoint train/calibration/test splits with every target class "
        "and OOD examples in calibration and test. Add more independent prompts and unknown generators."
    )


def _expected_calibration_error(probabilities: np.ndarray, encoded_labels: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, 11)
    error = 0.0
    for index in range(len(edges) - 1):
        in_bin = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == len(edges) - 2 else confidence < edges[index + 1]
        )
        if in_bin.any():
            error += float(in_bin.mean()) * abs(
                float((predicted[in_bin] == encoded_labels[in_bin]).mean())
                - float(confidence[in_bin].mean())
            )
    return error


def train_research_model(
    features_path: str | Path,
    model_out: str | Path,
    report_dir: str | Path,
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Train, calibrate, evaluate, and package a CommunityForensics model."""
    features_path = Path(features_path)
    model_out = Path(model_out)
    report_dir = Path(report_dir)
    required = {
        "label", "is_ood", "group_id", "flat_spectrum", "visual_features", "scalar_12", "dataset_kind"
    }
    optional_columns = {
        "sample_kind",
        "source_shard",
        "source_split",
        "source_label",
        "subset",
        "model_name",
        "real_source",
    }
    available = set(pq.read_schema(features_path).names)
    missing = required - available
    if missing:
        raise ValueError(f"Feature file is missing required columns: {', '.join(sorted(missing))}")
    frame = pd.read_parquet(features_path, columns=sorted(required | (available & optional_columns)))
    if set(frame["dataset_kind"].dropna().unique()) != {"communityforensics"}:
        raise ValueError("Research training accepts only CommunityForensics features; demo data is not eligible")
    if not set(TARGET_LABELS).issubset(set(frame.loc[~frame["is_ood"], "label"])):
        raise ValueError("Feature file does not contain all configured target classes")

    train_idx, calibration_idx, test_idx = _select_group_splits(frame, random_state)
    train_rows = frame.iloc[train_idx]
    calibration_rows = frame.iloc[calibration_idx]
    test_rows = frame.iloc[test_idx]
    train_known = train_rows.loc[~train_rows["is_ood"] & train_rows["label"].isin(TARGET_LABELS)]
    calibration_known = calibration_rows.loc[
        ~calibration_rows["is_ood"] & calibration_rows["label"].isin(TARGET_LABELS)
    ]
    calibration_ood = calibration_rows.loc[calibration_rows["is_ood"]]
    test_known = test_rows.loc[~test_rows["is_ood"] & test_rows["label"].isin(TARGET_LABELS)]
    test_ood = test_rows.loc[test_rows["is_ood"]]
    if len(calibration_ood) < 2 or len(test_ood) < 2:
        raise ValueError("Need at least two OOD examples in both calibration and held-out test groups")

    extractor = FeatureExtractor(n_pca_components=128)
    train_spectra = np.stack(train_known["flat_spectrum"].map(lambda value: np.asarray(value, dtype=np.float32)))
    extractor.fit_pca(train_spectra)
    if extractor.pca.n_components_ != 128:
        raise ValueError("At least 129 independent real training rows are needed for 128 spectral PCA features")
    X_train = _hybrid_features(train_known, extractor)
    y_train = train_known["label"].astype(str).to_numpy()

    classifier = AttributionClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    # Cross-validation refits PCA and the classifier inside every group fold.
    groups_train = train_known["group_id"].astype(str).to_numpy()
    strata_train = y_train
    per_class_groups = train_known.groupby("label")["group_id"].nunique()
    fold_count = min(cv_folds, int(per_class_groups.min()), len(np.unique(groups_train)))
    if fold_count < 2:
        raise ValueError("Need at least two independent prompt groups per training split")
    cv = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=random_state)
    cv_scores: list[float] = []
    for fold_train, fold_valid in cv.split(X_train, strata_train, groups_train):
        fold_extractor = FeatureExtractor(n_pca_components=128)
        fold_raw = np.stack(
            train_known.iloc[fold_train]["flat_spectrum"].map(
                lambda value: np.asarray(value, dtype=np.float32)
            )
        )
        fold_extractor.fit_pca(fold_raw)
        fold_model = AttributionClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        fold_model.fit(
            _hybrid_features(train_known.iloc[fold_train], fold_extractor), y_train[fold_train]
        )
        fold_predictions = fold_model.predict(
            _hybrid_features(train_known.iloc[fold_valid], fold_extractor)
        )
        cv_scores.append(float(f1_score(y_train[fold_valid], fold_predictions, average="macro")))

    classifier.fit(X_train, y_train)
    X_cal = _hybrid_features(calibration_known, extractor)
    y_cal = calibration_known["label"].astype(str).to_numpy()
    temperature = classifier.calibrate_temperature(X_cal, y_cal)
    X_cal_ood = _hybrid_features(calibration_ood, extractor)
    ood_calibration = classifier.calibrate_ood(X_cal, X_cal_ood)

    X_test_known = _hybrid_features(test_known, extractor)
    y_test = test_known["label"].astype(str).to_numpy()
    test_results = classifier.evaluate(X_test_known, y_test)
    test_probs = classifier.predict_proba(X_test_known)
    encoded = classifier.label_encoder.transform(y_test)
    test_results["expected_calibration_error"] = _expected_calibration_error(test_probs, encoded)
    X_test_ood = _hybrid_features(test_ood, extractor)
    test_ood_results = [classifier.predict_single(vector) for vector in X_test_ood]
    ood_probabilities = np.asarray([item["ood_probability"] for item in test_ood_results])
    robustness_results: dict[str, Any] = {}
    for kind in ("severe_jpeg", "severe_downscale", "severe_blur"):
        positions = np.flatnonzero(test_ood["sample_kind"].to_numpy() == kind)
        if len(positions):
            expected = test_ood.iloc[positions]["label"].astype(str).to_numpy()
            predicted = np.asarray([test_ood_results[int(i)]["predicted_engine"] for i in positions])
            uncertain = np.asarray([test_ood_results[int(i)]["is_ood"] for i in positions])
            robustness_results[kind] = {
                "samples": int(len(positions)),
                "top_class_accuracy": float(accuracy_score(expected, predicted)),
                "uncertain_rate": float(np.mean(uncertain)),
            }
    ood_results = {
        "held_out_ood_recall": float(np.mean(ood_probabilities > classifier.ood_threshold)),
        "held_out_ood_auroc": float(
            roc_auc_score(
                np.concatenate([np.zeros(len(X_test_known)), np.ones(len(X_test_ood))]),
                np.concatenate(
                    [
                        classifier.ood_detector.predict_proba(classifier._ood_features(X_test_known))[:, 1],
                        ood_probabilities,
                    ]
                ),
            )
        ),
        "held_out_ood_count": int(len(test_ood)),
        "severe_degradation": robustness_results,
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_out.with_suffix(".manifest.json")
    manifest_path.unlink(missing_ok=True)
    classifier.save(model_out)
    extractor_path = model_out.parent / "feature_extractor.pkl"
    extractor.save(extractor_path)
    dataset_digest = hashlib.sha256()
    with features_path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            dataset_digest.update(chunk)
    manifest = {
        "format_version": 1,
        "model_type": "hybrid_524d",
        "dataset_id": DATASET_ID,
        "dataset_kind": "communityforensics",
        "dataset_sha256": dataset_digest.hexdigest(),
        "evaluation_status": "research_holdout_complete",
        "evaluated_on_heldout_groups": True,
        "grouping_method": GROUPING_METHOD,
        "grouping_limitations": [
            "The Parquet schema has no generation-batch ID or individual source-image ID; real_source names a dataset, not a unique source image."
        ],
        "classes": list(classifier.class_names),
        "feature_extractor": extractor_path.name,
        "preprocessing": {"crop_size": 512, "color": "RGB", "crop_policy": "center-reflect-pad"},
        "visual_model": "vit_small_patch14_dinov2.lvd142m",
        "probabilities_calibrated": classifier.probabilities_calibrated,
        "temperature": temperature,
        "ood_calibration": ood_calibration,
        "ood_threshold": classifier.ood_threshold,
        "created_at": datetime.now(UTC).isoformat(),
        "sample_counts": {
            "train_known": int(len(train_known)),
            "calibration_known": int(len(calibration_known)),
            "calibration_ood": int(len(calibration_ood)),
            "test_known": int(len(test_known)),
            "test_ood": int(len(test_ood)),
            "train_groups": int(train_known["group_id"].nunique()),
            "calibration_groups": int(calibration_rows["group_id"].nunique()),
            "test_groups": int(test_rows["group_id"].nunique()),
        },
        "source_provenance": {
            column: sorted(str(value) for value in frame[column].dropna().unique())
            for column in (
                "source_shard",
                "source_split",
                "source_label",
                "subset",
                "model_name",
                "real_source",
            )
            if column in frame.columns
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {
        "test": test_results,
        "cv": {"macro_f1_mean": float(np.mean(cv_scores)), "macro_f1_std": float(np.std(cv_scores)), "fold_scores": cv_scores},
        "ood": ood_results,
        "calibration": {"temperature": temperature, **ood_calibration},
        "sample_counts": manifest["sample_counts"],
        "model_manifest": str(manifest_path),
    }
    (report_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    classifier.feature_importance(50).to_csv(report_dir / "feature_importance.csv", index=False)
    return result

"""Prompt-grouped training, calibration, OOD scoring, and held-out evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
import joblib
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import IMAGE_EXPORT_PER_CLASS, PCA_COMPONENTS, TARGET_LABELS
from image_export import write_image_export


def select_splits(rows: list[dict[str, object]], seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([str(row["label"]) for row in rows])
    is_ood = np.asarray([bool(row["is_ood"]) for row in rows])
    groups = np.asarray([str(row["group_id"]) for row in rows])
    strata = np.where(is_ood, "ood", labels)
    counts = [len(np.unique(groups[strata == value])) for value in np.unique(strata)]
    folds = min(5, min(counts))
    if folds < 3:
        raise ValueError("Need at least three independent prompt groups per class/OOD stratum")
    outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for development, test in outer.split(np.zeros(len(rows)), strata, groups):
        if not set(TARGET_LABELS).issubset(set(labels[test][~is_ood[test]])) or not is_ood[test].any():
            continue
        inner_strata, inner_groups = strata[development], groups[development]
        inner_counts = [len(np.unique(inner_groups[inner_strata == value])) for value in np.unique(inner_strata)]
        inner_folds = min(5, min(inner_counts))
        if inner_folds < 2:
            continue
        inner = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed + 1)
        for train_local, calibration_local in inner.split(np.zeros(len(development)), inner_strata, inner_groups):
            train, calibration = development[train_local], development[calibration_local]
            if (
                set(TARGET_LABELS).issubset(set(labels[train][~is_ood[train]]))
                and set(TARGET_LABELS).issubset(set(labels[calibration][~is_ood[calibration]]))
                and is_ood[calibration].any()
            ):
                partitions = (train, calibration, test)
                group_sets = [set(groups[indices]) for indices in partitions]
                if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
                    raise RuntimeError("Prompt groups crossed train, calibration, and test partitions")
                known_partition_indices = [indices[~is_ood[indices]] for indices in partitions]
                minimums = (2, 2, IMAGE_EXPORT_PER_CLASS)
                for partition_name, indices, minimum in zip(
                    ("train", "calibration", "test"), known_partition_indices, minimums
                ):
                    counts = {label: int(np.sum(labels[indices] == label)) for label in TARGET_LABELS}
                    insufficient = {label: count for label, count in counts.items() if count < minimum}
                    if insufficient:
                        raise ValueError(
                            f"{partition_name} partition needs at least {minimum} clean samples per "
                            f"target class; insufficient counts: {insufficient}"
                        )
                return train, calibration, test
    raise ValueError("Could not create prompt-group-disjoint train/calibration/test splits")

def hybrid_features(rows: list[dict[str, object]], indices: np.ndarray, scaler, pca) -> np.ndarray:
    raw = np.stack([rows[int(i)]["flat_spectrum"] for i in indices]).astype(np.float32)
    pca_values = pca.transform(scaler.transform(raw)).astype(np.float32)
    visual = np.stack([rows[int(i)]["visual_features"] for i in indices]).astype(np.float32)
    scalars = np.stack([rows[int(i)]["scalar_12"] for i in indices]).astype(np.float32)
    return np.concatenate((visual, pca_values, scalars), axis=1)

def model_probabilities(model, X: np.ndarray, temperature: float) -> np.ndarray:
    margins = np.asarray(model.predict(X, output_margin=True))
    if margins.ndim == 1:
        margins = margins.reshape(1, -1)
    logits = margins / max(temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)

def ood_features(model, X: np.ndarray, temperature: float, centroids: dict[str, np.ndarray], probabilities=None) -> np.ndarray:
    probabilities = probabilities if probabilities is not None else model_probabilities(model, X, temperature)
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1) / np.log(max(probabilities.shape[1], 2))
    margins = np.asarray(model.predict(X, output_margin=True))
    if margins.ndim == 1:
        margins = margins.reshape(1, -1)
    scaled = margins / max(temperature, 1e-6)
    max_logits = scaled.max(axis=1)
    energy = -(max_logits + np.log(np.exp(scaled - max_logits[:, None]).sum(axis=1)))
    similarities = []
    for vector in X[:, :384]:
        unit = vector / (np.linalg.norm(vector) + 1e-12)
        similarities.append(max(float(np.dot(unit, center)) for center in centroids.values()))
    return np.column_stack((confidence, entropy, similarities, energy))

def calibrate_and_evaluate(rows: list[dict[str, object]], output: Path) -> dict[str, object]:
    train_idx, calibration_idx, test_idx = select_splits(rows)
    y = np.asarray([str(row["label"]) for row in rows])
    ood = np.asarray([bool(row["is_ood"]) for row in rows])
    train_known = train_idx[~ood[train_idx] & np.isin(y[train_idx], TARGET_LABELS)]
    calibration_known = calibration_idx[~ood[calibration_idx] & np.isin(y[calibration_idx], TARGET_LABELS)]
    calibration_ood = calibration_idx[ood[calibration_idx]]
    test_known = test_idx[~ood[test_idx] & np.isin(y[test_idx], TARGET_LABELS)]
    test_ood = test_idx[ood[test_idx]]
    if len(calibration_ood) < 2 or len(test_ood) < 2:
        raise ValueError("Need at least two OOD examples in calibration and test groups")

    raw_train = np.stack([rows[int(i)]["flat_spectrum"] for i in train_known]).astype(np.float32)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(raw_train)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42, svd_solver="randomized")
    pca.fit(scaled_train)
    X_train = hybrid_features(rows, train_known, scaler, pca)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y[train_known])
    model_params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 4,
        "tree_method": "hist",
        "verbosity": 0,
        "num_class": len(np.unique(y_train)),
    }
    model = xgb.XGBClassifier(
        **model_params,
    )
    model.fit(X_train, y_train)
    centroids = {}
    for class_name in encoder.classes_:
        vectors = X_train[y[train_known] == class_name, :384]
        center = vectors.mean(axis=0)
        centroids[str(class_name)] = center / (np.linalg.norm(center) + 1e-12)

    X_cal = hybrid_features(rows, calibration_known, scaler, pca)
    y_cal = encoder.transform(y[calibration_known])
    margins = np.asarray(model.predict(X_cal, output_margin=True))
    if margins.ndim == 1:
        margins = margins.reshape(1, -1)

    def nll(temperature: float) -> float:
        logits = margins / temperature
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(-np.mean(log_probs[np.arange(len(y_cal)), y_cal]))

    temperature = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
    X_cal_ood = hybrid_features(rows, calibration_ood, scaler, pca)
    prob_cal = model_probabilities(model, X_cal, temperature)
    prob_cal_ood = model_probabilities(model, X_cal_ood, temperature)
    features_in = ood_features(model, X_cal, temperature, centroids, prob_cal)
    features_out = ood_features(model, X_cal_ood, temperature, centroids, prob_cal_ood)
    detector_X = np.vstack((features_in, features_out))
    detector_y = np.r_[np.zeros(len(features_in)), np.ones(len(features_out))]
    detector = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    )
    detector.fit(detector_X, detector_y)
    in_scores = detector.predict_proba(features_in)[:, 1]
    threshold = float(np.quantile(in_scores, 0.95, method="higher"))

    X_test = hybrid_features(rows, test_known, scaler, pca)
    y_test = y[test_known]
    test_probs = model_probabilities(model, X_test, temperature)
    predictions = encoder.inverse_transform(test_probs.argmax(axis=1))
    X_test_ood = hybrid_features(rows, test_ood, scaler, pca)
    test_ood_scores = detector.predict_proba(ood_features(model, X_test_ood, temperature, centroids))[:, 1]
    result = {
        "test": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "per_class": {},
            "known_samples": int(len(test_known)),
            "ood_samples": int(len(test_ood)),
            "ood_recall": float(np.mean(test_ood_scores > threshold)),
            "ood_threshold": threshold,
        },
        "calibration": {
            "temperature": temperature,
            "validation_false_reject_rate": float(np.mean(in_scores > threshold)),
            "validation_ood_recall": float(np.mean(detector.predict_proba(features_out)[:, 1] > threshold)),
            "validation_ood_auroc": float(roc_auc_score(detector_y, detector.predict_proba(detector_X)[:, 1])),
        },
        "split_counts": {
            "train_known": int(len(train_known)),
            "calibration_known": int(len(calibration_known)),
            "calibration_ood": int(len(calibration_ood)),
            "test_known": int(len(test_known)),
            "test_ood": int(len(test_ood)),
            "train_groups": len(set(str(rows[int(i)]["group_id"]) for i in train_known)),
            "calibration_groups": len(set(str(rows[int(i)]["group_id"]) for i in calibration_idx)),
            "test_groups": len(set(str(rows[int(i)]["group_id"]) for i in test_idx)),
        },
    }
    for class_name in TARGET_LABELS:
        mask = y_test == class_name
        result["test"]["per_class"][class_name] = {
            "samples": int(mask.sum()),
            "accuracy": float(np.mean(predictions[mask] == class_name)) if mask.any() else None,
            "f1": float(f1_score(y_test, predictions, labels=[class_name], average="macro", zero_division=0)),
        }

    image_manifest, image_counts = write_image_export(rows, test_idx)
    result["manual_test_images"] = {
        "manifest": image_manifest.name,
        "per_class": image_counts,
        "total": int(sum(image_counts.values())),
    }

    model_path = output / "attribution_hybrid.json"
    model.save_model(model_path)
    joblib.dump(
        {
            "label_encoder": encoder,
            "class_names": encoder.classes_.tolist(),
            "n_classes": len(encoder.classes_),
            "params": model_params,
            "visual_centroids": centroids,
            "temperature": temperature,
            "probabilities_calibrated": True,
            "ood_detector": detector,
            "ood_threshold": threshold,
        },
        output / "attribution_hybrid.pkl",
    )
    joblib.dump(
        {
            "pca": pca,
            "scaler": scaler,
            "n_pca_components": PCA_COMPONENTS,
            "n_subbands": 8,
            "r_max": 255,
        },
        output / "feature_extractor.pkl",
    )
    importance_path = output / "feature_importance.csv"
    with importance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_index", "importance"])
        writer.writerows(enumerate(np.asarray(model.feature_importances_, dtype=float)))
    (output / "evaluation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

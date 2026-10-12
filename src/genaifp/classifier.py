"""Module 4: Attribution Classifier.

Trains XGBoost multiclass classifier on PCA features.
Includes cross-validation, per-class metrics, and model persistence.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AttributionClassifier:
    """XGBoost-based multiclass attribution classifier."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        eval_metric: str = "mlogloss",
        early_stopping_rounds: int | None = None,
        **kwargs,
    ):
        """Initialize XGBoost classifier with sensible defaults."""
        self.early_stopping_rounds = early_stopping_rounds
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "eval_metric": eval_metric,
            "objective": "multi:softprob",
            "verbosity": 0,
        }
        self.params.update(kwargs)

        self.model: xgb.XGBClassifier | None = None
        self.label_encoder: LabelEncoder | None = None
        self.class_names: list[str] | None = None
        self.n_classes: int | None = None
        self._fitted = False
        self.temperature = 1.0
        self.probabilities_calibrated = False
        self.ood_detector: Any | None = None
        self.ood_threshold = 0.5

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
    ) -> AttributionClassifier:
        """Train the classifier.

        Args:
            X: (N, D) PCA features
            y: (N,) class labels (strings or ints)
            X_val, y_val: optional validation set for early stopping
            early_stopping_rounds: early stopping patience
            verbose: print training progress
        """
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_.tolist()
        self.n_classes = len(self.class_names)
        self.temperature = 1.0
        self.probabilities_calibrated = False
        self.ood_detector = None

        # Compute normalized visual centroids for OOD distance verification
        self.visual_centroids = {}
        if X.shape[1] == 524:
            X_vis = X[:, :384]
            for cls in self.class_names:
                cls_mask = y == cls
                if np.any(cls_mask):
                    c = np.mean(X_vis[cls_mask], axis=0)
                    self.visual_centroids[cls] = c / (np.linalg.norm(c) + 1e-12)

        self.params["num_class"] = self.n_classes

        # Build model with optional early stopping in constructor (XGBoost 2.0+ compatible)
        xgb_params = dict(self.params)
        stop_rounds = (
            early_stopping_rounds
            if early_stopping_rounds is not None
            else self.early_stopping_rounds
        )
        if X_val is not None and y_val is not None and stop_rounds is not None:
            xgb_params["early_stopping_rounds"] = stop_rounds

        self.model = xgb.XGBClassifier(**xgb_params)

        # Fit params
        fit_params = {}
        if X_val is not None and y_val is not None:
            y_val_enc = self.label_encoder.transform(y_val)
            fit_params["eval_set"] = [(X_val, y_val_enc)]
            fit_params["verbose"] = verbose

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y_enc, **fit_params)

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Returns:
            (N, n_classes) probability matrix
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        margins = np.asarray(self.model.predict(X, output_margin=True))
        if margins.ndim == 1:
            margins = margins.reshape(1, -1)
        logits = margins / max(float(self.temperature), 1e-6)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def calibrate_temperature(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fit one temperature on held-out in-distribution calibration samples."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        from scipy.optimize import minimize_scalar

        y_encoded = self.label_encoder.transform(y)
        margins = np.asarray(self.model.predict(X, output_margin=True))
        if margins.ndim == 1:
            margins = margins.reshape(1, -1)

        def nll(temperature: float) -> float:
            logits = margins / temperature
            logits -= np.max(logits, axis=1, keepdims=True)
            log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            return float(-np.mean(log_probs[np.arange(len(y_encoded)), y_encoded]))

        result = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        self.temperature = float(result.x)
        self.probabilities_calibrated = True
        return self.temperature

    def _ood_features(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        confidence = probabilities.max(axis=1)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
        normalized_entropy = entropy / max(np.log(max(len(self.class_names), 2)), 1e-12)
        energy = []
        similarity = []
        margins = np.asarray(self.model.predict(X, output_margin=True))
        if margins.ndim == 1:
            margins = margins.reshape(1, -1)
        for row in margins:
            scaled = row / max(float(self.temperature), 1e-6)
            maximum = float(np.max(scaled))
            energy.append(-(maximum + np.log(np.exp(scaled - maximum).sum())))
        if X.shape[1] == 524 and getattr(self, "visual_centroids", None):
            for vector in X[:, :384]:
                normalized = vector / (np.linalg.norm(vector) + 1e-12)
                similarity.append(max(float(np.dot(normalized, c)) for c in self.visual_centroids.values()))
        else:
            similarity = [0.0] * len(X)
        return np.column_stack([confidence, normalized_entropy, similarity, np.asarray(energy)])

    def calibrate_ood(self, X_in: np.ndarray, X_ood: np.ndarray) -> dict[str, float]:
        """Fit an OOD score model and cap validation false rejections at 5%."""
        if len(X_in) < 2 or len(X_ood) < 2:
            raise ValueError("OOD calibration needs at least two in-distribution and two OOD samples")
        features_in = self._ood_features(X_in)
        features_ood = self._ood_features(X_ood)
        features = np.vstack([features_in, features_ood])
        labels = np.concatenate([np.zeros(len(features_in)), np.ones(len(features_ood))])
        detector = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        )
        detector.fit(features, labels)
        in_scores = detector.predict_proba(features_in)[:, 1]
        self.ood_threshold = float(np.quantile(in_scores, 0.95, method="higher"))
        self.ood_detector = detector
        out_scores = detector.predict_proba(features_ood)[:, 1]
        return {
            "validation_false_reject_rate": float(np.mean(in_scores > self.ood_threshold)),
            "validation_ood_recall": float(np.mean(out_scores > self.ood_threshold)),
            "validation_ood_auroc": float(roc_auc_score(labels, detector.predict_proba(features)[:, 1])),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (as strings)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        y_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_single(self, x: np.ndarray, temperature: float | None = None) -> dict[str, Any]:
        """Predict single sample with calibrated confidence, entropy, and OOD detection.

        Returns dict matching spec's predict.py JSON format.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        x = x.reshape(1, -1)
        scoring_temperature = max(
            float(self.temperature if temperature is None else temperature), 1e-6
        )
        probs = self.predict_proba(x)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # Calculate Shannon entropy: H = -sum(p * log(p))
        eps = 1e-12
        entropy = float(-np.sum(probs * np.log(probs + eps)))
        max_entropy = float(np.log(len(self.class_names)))
        norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        # Energy-based OOD score: E(x) = -T * log(sum(exp(f_c(x)/T))).
        try:
            margins = self.model.predict(x, output_margin=True)[0]
            max_m = np.max(margins / scoring_temperature)
            lse = max_m + np.log(np.sum(np.exp(margins / scoring_temperature - max_m)))
            energy = float(-scoring_temperature * lse)
        except Exception:
            energy = float(-np.log(confidence + eps))

        # Cosine similarity to nearest training visual centroid
        vis_similarity = None
        if hasattr(self, "visual_centroids") and self.visual_centroids and x.shape[1] == 524:
            x_vis = x[0, :384]
            x_vis_norm = x_vis / (np.linalg.norm(x_vis) + 1e-12)
            sims = [float(np.dot(x_vis_norm, c)) for c in self.visual_centroids.values()]
            vis_similarity = float(max(sims))

        if self.ood_detector is not None:
            ood_probability = float(self.ood_detector.predict_proba(self._ood_features(x))[0, 1])
            is_ood = ood_probability > self.ood_threshold
        else:
            # Uncalibrated models can classify, but must not claim a calibrated OOD decision.
            ood_probability = None
            is_ood = True

        out = {
            "predicted_engine": self.class_names[pred_idx],
            "confidence": confidence,
            "status": "confident" if not is_ood else "uncertain",
            "entropy": entropy,
            "normalized_entropy": norm_entropy,
            "energy_score": energy,
            "feature_similarity": vis_similarity,
            "is_ood": is_ood,
            "ood_probability": ood_probability,
            "probability_calibrated": self.probabilities_calibrated,
            "class_probabilities": {
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            },
        }
        if is_ood:
            reasons = []
            if ood_probability is None:
                reasons.append("no held-out OOD calibration is available")
            else:
                reasons.append(f"calibrated OOD score {ood_probability:.2f} exceeds its threshold")
            out["warning"] = "Uncertain attribution: " + "; ".join(reasons) + "."
        return out

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Full evaluation on test set."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        y_enc = self.label_encoder.transform(y)
        y_pred = self.model.predict(X)
        self.predict_proba(X)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_enc, y_pred, average=None, zero_division=0
        )

        report = classification_report(
            y_enc, y_pred, target_names=self.class_names, output_dict=True, zero_division=0
        )

        cm = confusion_matrix(y_enc, y_pred)

        return {
            "accuracy": float(accuracy_score(y_enc, y_pred)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "per_class": {
                name: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, name in enumerate(self.class_names)
            },
            "confusion_matrix": cm.tolist(),
            "class_names": self.class_names,
            "brier_score": float(
                np.mean(
                    [
                        brier_score_loss((y_enc == i).astype(int), self.predict_proba(X)[:, i])
                        for i in range(len(self.class_names))
                    ]
                )
            ),
        }

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5, random_state: int = 42
    ) -> dict[str, Any]:
        """Stratified k-fold cross-validation."""
        y_enc = (
            self.label_encoder.transform(y)
            if self.label_encoder
            else LabelEncoder().fit_transform(y)
        )

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        # We need to refit per fold, so use cross_val_score on a fresh model
        cv_params = dict(self.params)
        if self.n_classes is not None:
            cv_params["num_class"] = self.n_classes
        cv_model = xgb.XGBClassifier(**cv_params)

        scores = cross_val_score(cv_model, X, y_enc, cv=skf, scoring="f1_macro", n_jobs=-1)

        # Also get per-fold predictions for confusion matrix
        from sklearn.model_selection import cross_val_predict

        y_pred_cv = cross_val_predict(cv_model, X, y_enc, cv=skf, n_jobs=-1)
        cm_cv = confusion_matrix(y_enc, y_pred_cv)

        return {
            "cv_f1_macro_mean": float(scores.mean()),
            "cv_f1_macro_std": float(scores.std()),
            "cv_f1_scores": scores.tolist(),
            "cv_confusion_matrix": cm_cv.tolist(),
        }

    def save(self, path: str | Path) -> None:
        """Save model + label encoder to disk."""
        if not self._fitted:
            raise RuntimeError("Nothing to save - model not fitted")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(str(path.with_suffix(".json")))

        # Save label encoder + metadata
        meta = {
            "label_encoder": self.label_encoder,
            "class_names": self.class_names,
            "n_classes": self.n_classes,
            "params": self.params,
            "visual_centroids": getattr(self, "visual_centroids", {}),
            "temperature": self.temperature,
            "probabilities_calibrated": self.probabilities_calibrated,
            "ood_detector": self.ood_detector,
            "ood_threshold": self.ood_threshold,
        }
        joblib.dump(meta, path.with_suffix(".pkl"))

    @classmethod
    def load(cls, path: str | Path) -> AttributionClassifier:
        """Load model from disk."""
        path = Path(path)
        meta = joblib.load(path.with_suffix(".pkl"))

        clf = cls(**meta["params"])
        clf.model = xgb.XGBClassifier(**meta["params"])
        clf.model.load_model(str(path.with_suffix(".json")))
        clf.label_encoder = meta["label_encoder"]
        clf.class_names = meta["class_names"]
        clf.n_classes = meta["n_classes"]
        clf.visual_centroids = meta.get("visual_centroids", {})
        clf.temperature = float(meta.get("temperature", 1.0))
        clf.probabilities_calibrated = bool(meta.get("probabilities_calibrated", False))
        clf.ood_detector = meta.get("ood_detector")
        clf.ood_threshold = float(meta.get("ood_threshold", 0.5))
        clf._fitted = True
        return clf

    def feature_importance(self, top_k: int = 50) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        imp = self.model.feature_importances_
        n_feats = len(imp)

        if n_feats == 524:
            names = (
                [f"DINO_{i}" for i in range(384)]
                + [f"Spectral_PC{i}" for i in range(128)]
                + ["PASR", "Slope_Alpha", "Intercept_Beta"]
                + [f"SubBand_E{i + 1}" for i in range(8)]
                + ["Directional_Variance"]
            )
        elif n_feats == 128:
            names = [f"Spectral_PC{i}" for i in range(128)]
        elif n_feats == 140:
            names = (
                [f"Spectral_PC{i}" for i in range(128)]
                + ["PASR", "Slope_Alpha", "Intercept_Beta"]
                + [f"SubBand_E{i + 1}" for i in range(8)]
                + ["Directional_Variance"]
            )
        else:
            names = [f"Feature_{i}" for i in range(n_feats)]

        indices = np.argsort(imp)[::-1][: min(top_k, n_feats)]

        return pd.DataFrame(
            {
                "feature": [names[i] for i in indices],
                "importance": imp[indices],
            }
        )


def train_attribution_model(
    features_path: str | Path,
    model_out: str | Path = "models/attribution_xgb",
    report_dir: str | Path = "reports",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    feature_col: str | None = None,
    classifier_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """End-to-end training pipeline.

    Args:
        features_path: parquet file with features and 'label' columns
        model_out: output path for model (without extension)
        report_dir: directory for evaluation reports
        test_size: hold-out test fraction
        random_state: seed for reproducibility
        cv_folds: cross-validation folds
        n_estimators: number of trees in XGBoost
        max_depth: max depth of trees
        learning_rate: learning rate
        feature_col: column name containing feature vectors (auto-detects hybrid_features or pca_features)
        classifier_kwargs: additional kwargs for AttributionClassifier

    Returns:
        Dictionary with evaluation results
    """
    # Load features
    df = pd.read_parquet(features_path)

    if feature_col is None:
        if "hybrid_features" in df.columns:
            feature_col = "hybrid_features"
        elif "pca_features" in df.columns:
            feature_col = "pca_features"
        else:
            candidates = [
                c
                for c in df.columns
                if c not in ("label", "model_name", "image_name", "source_shard")
            ]
            if candidates:
                feature_col = candidates[0]
            else:
                raise ValueError(f"No valid feature column found in {df.columns.tolist()}")

    print(f"Training on feature column: '{feature_col}'")
    X = np.array(df[feature_col].tolist(), dtype=np.float32)
    y = np.array(df["label"].tolist())

    print(f"Loaded {len(df)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train with validation split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state, stratify=y_train
    )

    clf_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": random_state,
    }
    if classifier_kwargs:
        clf_params.update(classifier_kwargs)

    clf = AttributionClassifier(**clf_params)
    clf.fit(X_tr, y_tr, X_val, y_val, early_stopping_rounds=50, verbose=False)

    # Evaluate
    test_results = clf.evaluate(X_test, y_test)
    cv_results = clf.cross_validate(X_train, y_train, cv=cv_folds, random_state=random_state)

    # Combine results
    results = {
        "test": test_results,
        "cv": cv_results,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "classes": clf.class_names,
    }

    # Save model
    clf.save(model_out)

    # Check if feature extractor exists in features directory and copy to models dir
    feat_src = Path(features_path).parent / "feature_extractor.pkl"
    feat_dst = Path(model_out).parent / "feature_extractor.pkl"
    if feat_src.exists() and not feat_dst.exists():
        import shutil

        shutil.copy2(feat_src, feat_dst)

    # Save reports
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Confusion matrix as CSV
    cm_df = pd.DataFrame(
        test_results["confusion_matrix"], index=clf.class_names, columns=clf.class_names
    )
    cm_df.to_csv(report_dir / "confusion_matrix.csv")

    # Classification report as CSV
    cr_rows = []
    for cls, metrics in test_results["per_class"].items():
        row = {"class": cls, **metrics}
        cr_rows.append(row)
    pd.DataFrame(cr_rows).to_csv(report_dir / "classification_report.csv", index=False)

    # Feature importance
    fi = clf.feature_importance(50)
    fi.to_csv(report_dir / "feature_importance.csv", index=False)

    print("\n=== Test Results ===")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Macro F1: {test_results['macro_f1']:.4f}")
    print(
        f"CV Macro F1: {cv_results['cv_f1_macro_mean']:.4f} ± {cv_results['cv_f1_macro_std']:.4f}"
    )
    print(f"\nReports saved to: {report_dir}")
    print(f"Model saved to: {model_out}.json / .pkl")

    return results


if __name__ == "__main__":
    # Self-test with synthetic data
    import tempfile

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500,
        n_features=128,
        n_informative=50,
        n_redundant=10,
        n_classes=6,
        random_state=42,
    )
    y = np.array([f"class_{i}" for i in y])

    with tempfile.TemporaryDirectory() as td:
        synth_df = pd.DataFrame(
            {
                "pca_features": list(X),
                "label": y,
            }
        )
        pq_path = Path(td) / "features.parquet"
        synth_df.to_parquet(pq_path)

        results = train_attribution_model(
            features_path=pq_path,
            model_out=f"{td}/model",
            report_dir=f"{td}/reports",
            n_estimators=10,  # quick test
        )
        print("Self-test passed! Macro F1:", results["test"]["macro_f1"])

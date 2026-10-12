"""Tests for classifier module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from genaifp.classifier import AttributionClassifier, train_attribution_model


class TestAttributionClassifier:
    @pytest.fixture
    def synth_data(self):
        X, y = make_classification(
            n_samples=120, n_features=32, n_informative=20, n_classes=4, random_state=42
        )
        classes = ["dalle3", "midjourney_v6", "imagen3", "real_camera"]
        y_named = np.array([classes[i] for i in y])
        return X.astype(np.float32), y_named

    def test_fit_and_predict(self, synth_data):
        X, y = synth_data
        clf = AttributionClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X, y)
        assert clf._fitted
        assert len(clf.class_names) == 4

        preds = clf.predict(X[:5])
        assert len(preds) == 5
        assert all(p in clf.class_names for p in preds)

        probs = clf.predict_proba(X[:5])
        assert probs.shape == (5, 4)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_predict_single(self, synth_data):
        X, y = synth_data
        clf = AttributionClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X, y)

        single = clf.predict_single(X[0])
        assert "predicted_engine" in single
        assert "confidence" in single
        assert "class_probabilities" in single
        assert single["predicted_engine"] in clf.class_names
        assert 0.0 <= single["confidence"] <= 1.0

    def test_evaluate_and_cv(self, synth_data):
        X, y = synth_data
        clf = AttributionClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X[:80], y[:80])

        eval_res = clf.evaluate(X[80:], y[80:])
        assert "accuracy" in eval_res
        assert "macro_f1" in eval_res
        assert "confusion_matrix" in eval_res

        cv_res = clf.cross_validate(X[:80], y[:80], cv=3)
        assert "cv_f1_macro_mean" in cv_res
        assert "cv_f1_macro_std" in cv_res

    def test_save_and_load(self, synth_data):
        X, y = synth_data
        clf = AttributionClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X, y)
        orig_pred = clf.predict_single(X[0])

        with tempfile.TemporaryDirectory() as td:
            save_path = Path(td) / "test_model"
            clf.save(save_path)

            loaded = AttributionClassifier.load(save_path)
            assert loaded._fitted
            loaded_pred = loaded.predict_single(X[0])
            assert orig_pred["predicted_engine"] == loaded_pred["predicted_engine"]
            assert np.isclose(orig_pred["confidence"], loaded_pred["confidence"], atol=1e-5)

    def test_train_attribution_model_pipeline(self, synth_data):
        X, y = synth_data
        with tempfile.TemporaryDirectory() as td:
            df = pd.DataFrame(
                {
                    "pca_features": list(X),
                    "label": y,
                }
            )
            pq_path = Path(td) / "features.parquet"
            df.to_parquet(pq_path)

            # Create a mock feature_extractor.pkl in the same dir
            (Path(td) / "feature_extractor.pkl").touch()

            res = train_attribution_model(
                features_path=pq_path,
                model_out=f"{td}/models/test_model",
                report_dir=f"{td}/reports",
                test_size=0.25,
                cv_folds=3,
                n_estimators=10,
                max_depth=3,
            )

            assert "test" in res
            assert "cv" in res
            assert Path(f"{td}/reports/evaluation.json").exists()
            assert Path(f"{td}/models/test_model.json").exists()
            assert Path(f"{td}/models/test_model.pkl").exists()
            assert Path(f"{td}/models/feature_extractor.pkl").exists()

    def test_feature_importance_524d(self):
        # 524 dimensions
        X = np.random.randn(50, 524).astype(np.float32)
        y = np.array(["midjourney_v6", "dalle3"] * 25)
        clf = AttributionClassifier(n_estimators=10, max_depth=2, random_state=42)
        clf.fit(X, y)

        fi = clf.feature_importance(top_k=20)
        assert len(fi) == 20
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        # Check that semantic names were assigned
        first_feat = fi["feature"].iloc[0]
        assert any(
            first_feat.startswith(prefix)
            for prefix in ["DINO_", "Spectral_PC", "PASR", "Slope_", "SubBand_", "Directional_"]
        )

    def test_predict_single_ood_fields(self, synth_data):
        X, y = synth_data
        clf = AttributionClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X, y)
        res = clf.predict_single(X[0])

        assert "entropy" in res
        assert "normalized_entropy" in res
        assert "energy_score" in res
        assert "status" in res
        assert "is_ood" in res
        assert res["status"] in ("confident", "uncertain")

    def test_held_out_calibration_is_used_in_prediction(self, synth_data):
        X, y = synth_data
        clf = AttributionClassifier(n_estimators=15, max_depth=3, random_state=42)
        clf.fit(X[:80], y[:80])
        temperature = clf.calibrate_temperature(X[80:100], y[80:100])
        ood_metrics = clf.calibrate_ood(X[80:95], X[100:115])
        result = clf.predict_single(X[0])
        assert temperature > 0
        assert clf.probabilities_calibrated
        assert clf.ood_detector is not None
        assert "validation_ood_recall" in ood_metrics
        assert result["probability_calibrated"] is True
        assert result["ood_probability"] is not None
        margins = clf.model.predict(X[:1], output_margin=True)[0]
        scaled = margins / clf.temperature
        expected_energy = -clf.temperature * (
            np.max(scaled) + np.log(np.exp(scaled - np.max(scaled)).sum())
        )
        assert np.isclose(result["energy_score"], expected_energy)

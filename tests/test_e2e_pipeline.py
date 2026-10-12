"""End-to-end integration test for forensic attribution pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from genaifp.classifier import AttributionClassifier, train_attribution_model
from genaifp.feature_extractor import FeatureExtractor
from genaifp.preprocessor import preprocess_array, preprocess_image
from genaifp.spectral_engine import SpectralEngine


class TestE2EPipeline:
    def test_full_flow_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            models_dir = base_dir / "models"
            reports_dir = base_dir / "reports"
            samples_dir = base_dir / "samples"
            models_dir.mkdir()
            reports_dir.mkdir()
            samples_dir.mkdir()

            classes = ["dalle3", "midjourney_v6", "imagen3", "real_camera"]
            spec_engine = SpectralEngine(crop_size=512)

            # 1. Create synthetic images for 4 classes
            rng = np.random.default_rng(42)
            features_list = []
            flat_spectra = []

            for cls_idx, cls_name in enumerate(classes):
                for _img_i in range(15):
                    # Introduce subtle frequency signature per class
                    arr = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
                    if cls_idx == 0:
                        arr[::8, ::8] = 255  # grid pattern
                    elif cls_idx == 1:
                        arr[::4, ::4] = 200  # denser grid

                    residual = preprocess_array(arr, crop_size=512)
                    spectral = spec_engine.process(residual)
                    flat_spectra.append(spectral["flat_spectrum"])
                    features_list.append(
                        {
                            "label": cls_name,
                            "flat_spectrum": spectral["flat_spectrum"],
                        }
                    )

            # 2. Fit FeatureExtractor
            feat_ext = FeatureExtractor(n_pca_components=16)
            feat_ext.fit_pca(np.stack(flat_spectra))

            feat_path = models_dir / "feature_extractor.pkl"
            feat_ext.save(feat_path)

            # 3. Build parquet
            df_rows = []
            for row in features_list:
                pca_f = feat_ext.transform_pca(row["flat_spectrum"])
                df_rows.append(
                    {
                        "pca_features": pca_f,
                        "label": row["label"],
                    }
                )

            pq_path = base_dir / "features.parquet"
            pd.DataFrame(df_rows).to_parquet(pq_path)

            # 4. Train model
            train_results = train_attribution_model(
                features_path=pq_path,
                model_out=models_dir / "attribution_xgb",
                report_dir=reports_dir,
                test_size=0.2,
                cv_folds=2,
                n_estimators=10,
                max_depth=3,
            )

            assert "test" in train_results
            assert (models_dir / "attribution_xgb.json").exists()
            assert (models_dir / "attribution_xgb.pkl").exists()

            # 5. Save a suspect test image
            suspect_path = samples_dir / "suspect.png"
            suspect_arr = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
            suspect_arr[::8, ::8] = 255  # dalle3 signature
            Image.fromarray(suspect_arr).save(suspect_path)

            # 6. Run prediction on suspect image
            clf = AttributionClassifier.load(models_dir / "attribution_xgb")
            test_feat_ext = FeatureExtractor.load(feat_path)

            res_img = preprocess_image(suspect_path, allow_pad=True)
            spec_out = spec_engine.process(res_img)
            feats = test_feat_ext.extract(spec_out)

            pred = clf.predict_single(feats["pca_features"])
            assert "predicted_engine" in pred
            assert "confidence" in pred
            assert "class_probabilities" in pred
            assert pred["predicted_engine"] in classes
            assert len(pred["class_probabilities"]) == 4

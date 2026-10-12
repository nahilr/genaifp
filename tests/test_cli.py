"""Unit tests for the CLI entrypoint (main.py)."""

from __future__ import annotations

from pathlib import Path

from main import main, parse_build_args, parse_predict_args, parse_train_args


class TestCLI:
    def test_help_returns_zero(self, capsys):
        code = main(["--help"])
        assert code == 0
        captured = capsys.readouterr()
        assert "GenAI Fingerprint (genaifp)" in captured.out
        assert "predict" in captured.out
        assert "train" in captured.out

    def test_unknown_command_returns_one(self, capsys):
        code = main(["nonexistent_command"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Unknown command" in captured.err

    def test_arg_parsers(self):
        pred_args = parse_predict_args(["--image_path", "test.png", "--crop_size", "256"])
        assert pred_args.image_path == "test.png"
        assert pred_args.crop_size == 256

        train_args = parse_train_args(["--input", "features.parquet", "--cv_folds", "3"])
        assert train_args.input == "features.parquet"
        assert train_args.cv_folds == 3

        build_args = parse_build_args(["--demo", "--crop_size", "256"])
        assert build_args.demo is True
        assert build_args.crop_size == 256

    def test_predict_sample_image(self, capsys):
        sample = Path("samples/midjourney_v6_sample.png")
        if sample.exists():
            code = main(["predict", "--image_path", str(sample)])
            assert code == 0
            captured = capsys.readouterr()
            assert '"predicted_engine"' in captured.out

    def test_direct_image_path_shortcut(self, capsys):
        sample = Path("samples/midjourney_v6_sample.png")
        if sample.exists():
            code = main([str(sample)])
            assert code == 0
            captured = capsys.readouterr()
            assert '"predicted_engine"' in captured.out

    def test_build_without_parquet_shards_returns_error(self, tmp_path, capsys):
        code = main(["build", "--data_dir", str(tmp_path), "--output", str(tmp_path / "out.parquet")])
        assert code == 1
        assert "real-data feature extraction runs in Kaggle" in capsys.readouterr().err

    def test_demo_training_requires_explicit_flag(self, tmp_path, capsys):
        import pandas as pd
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=60, n_features=20, n_informative=10, n_classes=3, random_state=1
        )
        source = tmp_path / "demo.parquet"
        pd.DataFrame({"label": [str(label) for label in y], "pca_features": list(X)}).to_parquet(source)
        code = main(["train", "--input", str(source), "--model_out", str(tmp_path / "m")])
        assert code == 1
        assert "--demo-only" in capsys.readouterr().err

"""Tests for real-data import and provenance helpers."""

import hashlib
import io
import sys
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from genaifp.data_pipeline import (
    TARGET_MODELS,
    _group_id,
    _image_bytes,
    _normalise_model,
    build_communityforensics_features,
)


def test_image_bytes_accepts_dataset_payload_shapes():
    value = b"encoded-image"
    assert _image_bytes(value) == value
    assert _image_bytes({"bytes": value, "path": None}) == value
    assert _image_bytes(np.frombuffer(value, dtype=np.uint8)) == value


def test_group_id_uses_prompt_to_keep_related_samples_together():
    first = _group_id("a   shared prompt", b"image one")
    second = _group_id("A shared prompt", b"image two")
    assert first == second
    assert _group_id("", b"image one") == hashlib.sha256(
        ("image:" + hashlib.sha256(b"image one").hexdigest()).encode()
    ).hexdigest()


def test_generator_names_are_normalized_for_mapping():
    assert _normalise_model("OwensLab/MidjourneyV6_1") == "midjourneyv61"


def test_feature_builder_fails_when_no_shards_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_communityforensics_features(tmp_path, tmp_path / "features.parquet")


def test_feature_builder_filters_classes_and_preserves_provenance(monkeypatch, tmp_path):
    class DummyVisualEngine:
        def __init__(self, crop_size=512):
            self.crop_size = crop_size

        def extract(self, image):
            return np.ones(384, dtype=np.float32)

    fake_module = types.ModuleType("genaifp.visual_engine")
    fake_module.VisualEngine = DummyVisualEngine
    monkeypatch.setitem(sys.modules, "genaifp.visual_engine", fake_module)
    rows = []
    names = [
        "Dalle2", "Dalle3", "MidjourneyV5_2", "MidjourneyV6_1",
        "Firefly_Image2", "Firefly_Image3", "FLUX-dev", "FLUX-schnell",
        "IdeogramV1", "IdeogramV2", "Imagen3",
    ]
    unknown_models = ["OtherGenerator", "AnotherGenerator"]
    for model in [*names, "LandscapesHQ", *unknown_models]:
        count = 3 if model in unknown_models else 2
        for index in range(count):
            image_buffer = io.BytesIO()
            color = (60 + index, 100 + len(model), 140 + len(rows) % 100)
            Image.new("RGB", (40, 40), color).save(image_buffer, format="PNG")
            image_bytes = image_buffer.getvalue()
            rows.append(
                {
                    "image_name": f"{model}-{index}.png",
                    "image_data": image_bytes,
                    "model_name": model,
                    "prompt": f"unique prompt {model} {index}",
                    "real_source": "test-source" if model == "LandscapesHQ" else "",
                    "subset": "Commercial",
                    "split": "train",
                    "label": 0 if model == "LandscapesHQ" else 1,
                }
            )
    input_path = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows), input_path)
    output_path = tmp_path / "features.parquet"
    result = build_communityforensics_features(
        tmp_path,
        output_path,
        crop_size=32,
        max_per_class=2,
        max_unknown_per_model=2,
        max_unknown_total=3,
        include_degraded=False,
    )
    built = pq.read_table(output_path).to_pandas()
    assert result["target_counts"]["real_camera"] == 2
    assert result["unknown_counts"] == {"anothergenerator": 1, "othergenerator": 2}
    assert result["unknown_total"] == 3
    assert set(built["label"]) == {*TARGET_MODELS.values(), "real_camera", "unknown"}
    assert built["source_shard"].eq("shard.parquet").all()
    assert built["dataset_kind"].eq("communityforensics").all()
    assert built["image_data"].map(lambda value: isinstance(value, bytes)).all()
    assert built["subset"].eq("Commercial").all()
    assert built["source_label"].isin(["0", "1"]).all()

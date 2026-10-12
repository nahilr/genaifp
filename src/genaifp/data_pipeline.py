"""CommunityForensics import and feature preparation helpers."""

from __future__ import annotations

import hashlib
import io
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageFilter

from .feature_extractor import FeatureExtractor
from .preprocessor import center_crop_rgb, preprocess_array
from .spectral_engine import SpectralEngine

DATASET_ID = "OwensLab/CommunityForensics"
GROUPING_METHOD = "normalized_prompt_hash; exact-image-hash fallback when prompt is absent"
TARGET_MODELS = {
    "dalle2": "dalle2",
    "dalle3": "dalle3",
    "midjourneyv52": "midjourney_v5_2",
    "midjourneyv61": "midjourney_v6_1",
    "fireflyimage2": "firefly_image2",
    "fireflyimage3": "firefly_image3",
    "fluxdev": "flux_dev",
    "fluxschnell": "flux_schnell",
    "ideogramv1": "ideogram_v1",
    "ideogramv2": "ideogram_v2",
    "imagen3": "imagen3",
}
REAL_MODELS = {"landscapeshq", "coco", "ffhq", "vision"}
TARGET_LABELS = (*sorted(TARGET_MODELS.values()), "real_camera")

OUTPUT_SCHEMA = pa.schema(
    [
        ("label", pa.string()),
        ("is_ood", pa.bool_()),
        ("sample_kind", pa.string()),
        ("group_id", pa.string()),
        ("image_name", pa.string()),
        ("model_name", pa.string()),
        ("prompt", pa.string()),
        ("real_source", pa.string()),
        ("subset", pa.string()),
        ("source_shard", pa.string()),
        ("source_split", pa.string()),
        ("source_label", pa.string()),
        ("dataset_id", pa.string()),
        ("dataset_kind", pa.string()),
        ("visual_features", pa.list_(pa.float32())),
        ("flat_spectrum", pa.list_(pa.float32())),
        ("scalar_12", pa.list_(pa.float32())),
        ("image_data", pa.binary()),
    ]
)


def _normalise_model(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").split("/")[-1].lower())


def _image_bytes(value: Any) -> bytes:
    """Normalize Hugging Face/PyArrow image payloads into encoded bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, dict):
        if value.get("bytes"):
            return _image_bytes(value["bytes"])
        raise ValueError("image_data has no embedded bytes")
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.uint8).tobytes()
    raise ValueError(f"Unsupported image_data payload: {type(value).__name__}")


def _group_id(prompt: Any, image_data: bytes) -> str:
    normalized = " ".join(str(prompt or "").casefold().split())
    key = f"prompt:{normalized}" if normalized else "image:" + hashlib.sha256(image_data).hexdigest()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _feature_row(
    image_data: bytes,
    image_name: str,
    label: str,
    is_ood: bool,
    sample_kind: str,
    group_id: str,
    model_name: str,
    prompt: str,
    real_source: str,
    subset: str,
    source_shard: str,
    source_split: str,
    source_label: str,
    store_image_data: bool,
    visual_engine: Any,
    spectral_engine: SpectralEngine,
) -> dict[str, Any]:
    with Image.open(io.BytesIO(image_data)) as opened:
        opened.load()
        image = center_crop_rgb(np.asarray(opened.convert("RGB")), spectral_engine.crop_size)
    residual = preprocess_array(image, crop_size=spectral_engine.crop_size, allow_pad=False)
    spectrum = spectral_engine.process(residual)
    scalars = FeatureExtractor.extract_12_scalar(spectrum["rapsd"], spectrum["azimuthal"])
    visual = visual_engine.extract(image)
    return {
        "label": label,
        "is_ood": is_ood,
        "sample_kind": sample_kind,
        "group_id": group_id,
        "image_name": image_name,
        "model_name": model_name,
        "prompt": prompt,
        "real_source": real_source,
        "subset": subset,
        "source_shard": source_shard,
        "source_split": source_split,
        "source_label": source_label,
        "dataset_id": DATASET_ID,
        "dataset_kind": "communityforensics",
        # Keep NumPy buffers compact while building large datasets. Converting the
        # 131k-element spectrum to Python floats multiplies its transient RAM use.
        "visual_features": visual,
        "flat_spectrum": spectrum["flat_spectrum"],
        "scalar_12": scalars,
        "image_data": image_data if store_image_data else None,
    }


def _severe_degradation(image_data: bytes, index: int) -> tuple[bytes, str]:
    """Create one held-out degradation example per source image."""
    with Image.open(io.BytesIO(image_data)) as opened:
        image = opened.convert("RGB")
    if index % 3 == 0:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=18, optimize=True)
        return buffer.getvalue(), "severe_jpeg"
    if index % 3 == 1:
        width, height = image.size
        tiny = image.resize((max(8, width // 8), max(8, height // 8)), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        tiny.resize((width, height), Image.Resampling.BICUBIC).save(buffer, format="JPEG", quality=55)
        return buffer.getvalue(), "severe_downscale"
    buffer = io.BytesIO()
    image.filter(ImageFilter.GaussianBlur(radius=8)).save(buffer, format="PNG")
    return buffer.getvalue(), "severe_blur"


def build_communityforensics_features(
    data_dir: str | Path,
    output_path: str | Path,
    *,
    crop_size: int = 512,
    max_per_class: int = 500,
    max_unknown_per_model: int = 100,
    max_unknown_total: int = 1000,
    include_degraded: bool = True,
    batch_size: int = 1,
    write_batch_size: int = 8,
    store_image_data: bool = True,
) -> dict[str, Any]:
    """Stream local CommunityForensics Parquet shards into a provenance-rich feature file."""
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    shard_paths = sorted(data_dir.glob("*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No Parquet shards found in {data_dir}")
    if max_per_class < 2 or max_unknown_per_model < 1 or max_unknown_total < 1:
        raise ValueError("Sample limits must be positive (at least two target samples per class)")
    if batch_size < 1 or write_batch_size < 1:
        raise ValueError("Parquet read and write batch sizes must be positive")

    from .visual_engine import VisualEngine

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".part")
    partial_path.unlink(missing_ok=True)
    visual_engine = VisualEngine(crop_size=crop_size)
    spectral_engine = SpectralEngine(crop_size=crop_size)
    target_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    unknown_total = 0
    failures: list[str] = []
    seen_image_hashes: set[str] = set()
    writer: pq.ParquetWriter | None = None
    buffered: list[dict[str, Any]] = []
    degraded_index = 0

    def flush() -> None:
        nonlocal writer, buffered
        if not buffered:
            return
        table = pa.Table.from_pylist(buffered, schema=OUTPUT_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(partial_path, OUTPUT_SCHEMA, compression="zstd")
        writer.write_table(table)
        buffered = []

    wanted_columns = [
        "image_name", "image_data", "model_name", "prompt", "real_source", "subset", "split", "label"
    ]
    try:
        for shard_path in shard_paths:
            parquet = pq.ParquetFile(shard_path)
            available = set(parquet.schema_arrow.names)
            if not {"image_data", "model_name"}.issubset(available):
                failures.append(f"{shard_path.name}: missing image_data/model_name columns")
                continue
            columns = [column for column in wanted_columns if column in available]
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
                for record in batch.to_pylist():
                    raw_model = str(record.get("model_name") or "")
                    model_key = _normalise_model(raw_model)
                    if model_key in TARGET_MODELS:
                        label, is_ood = TARGET_MODELS[model_key], False
                        if target_counts[label] >= max_per_class:
                            continue
                    elif model_key in REAL_MODELS or record.get("label") in (0, "0"):
                        label, is_ood = "real_camera", False
                        if target_counts[label] >= max_per_class:
                            continue
                    elif record.get("label") in (1, "1") and model_key:
                        if (
                            unknown_counts[model_key] >= max_unknown_per_model
                            or unknown_total >= max_unknown_total
                        ):
                            continue
                        label, is_ood = "unknown", True
                    else:
                        continue
                    try:
                        encoded = _image_bytes(record["image_data"])
                        with Image.open(io.BytesIO(encoded)) as image:
                            image.verify()
                        image_hash = hashlib.sha256(encoded).hexdigest()
                        if image_hash in seen_image_hashes:
                            continue
                        seen_image_hashes.add(image_hash)
                        prompt = str(record.get("prompt") or "")
                        group = _group_id(prompt, encoded)
                        common = {
                            "image_name": str(record.get("image_name") or ""),
                            "label": label,
                            "group_id": group,
                            "model_name": raw_model,
                            "prompt": prompt,
                            "real_source": str(record.get("real_source") or ""),
                            "subset": str(record.get("subset") or ""),
                            "source_shard": shard_path.name,
                            "source_split": str(record.get("split") or ""),
                            "source_label": (
                                str(record["label"]) if record.get("label") is not None else ""
                            ),
                            "store_image_data": store_image_data,
                            "visual_engine": visual_engine,
                            "spectral_engine": spectral_engine,
                        }
                        buffered.append(
                            _feature_row(
                                image_data=encoded,
                                is_ood=is_ood,
                                sample_kind="unsupported_generator" if is_ood else "clean",
                                **common,
                            )
                        )
                        if is_ood:
                            unknown_counts[model_key] += 1
                            unknown_total += 1
                        else:
                            target_counts[label] += 1
                        if include_degraded and not is_ood:
                            degraded, degraded_kind = _severe_degradation(encoded, degraded_index)
                            degraded_index += 1
                            buffered.append(
                                _feature_row(
                                    image_data=degraded,
                                    is_ood=True,
                                    sample_kind=degraded_kind,
                                    **common,
                                )
                            )
                        if len(buffered) >= write_batch_size:
                            flush()
                    except Exception as exc:
                        failures.append(f"{shard_path.name}/{record.get('image_name', '?')}: {exc}")
        flush()
    except Exception:
        if writer is not None:
            writer.close()
        partial_path.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.close()

    if writer is None:
        partial_path.unlink(missing_ok=True)
        raise ValueError("No usable target or unsupported-generator image rows were found")
    missing = [name for name in TARGET_LABELS if target_counts[name] < 2]
    if missing:
        partial_path.unlink(missing_ok=True)
        raise ValueError(f"Insufficient usable samples for classes: {', '.join(missing)}")
    if sum(unknown_counts.values()) < 2:
        partial_path.unlink(missing_ok=True)
        raise ValueError("Need unsupported-generator samples to calibrate the OOD rejection threshold")
    partial_path.replace(output_path)
    return {
        "dataset_id": DATASET_ID,
        "output": str(output_path),
        "target_counts": dict(target_counts),
        "unknown_counts": dict(unknown_counts),
        "unknown_total": unknown_total,
        "degraded_examples": degraded_index,
        "malformed_rows": len(failures),
        "failure_examples": failures[:20],
    }

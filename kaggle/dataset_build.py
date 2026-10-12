"""Scan source shards, apply sample caps, export examples, and write feature Parquet."""
from __future__ import annotations

import hashlib
import io
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from config import IMAGE_CANDIDATE_DIR, MAX_PER_CLASS, MAX_UNKNOWN_PER_MODEL, MAX_UNKNOWN_TOTAL, OUTPUT_DIR, RAW_DIR, SHARDS, TARGET_LABELS
from data_access import download_shard, group_id, image_bytes, label_record, normalize_model
from features import build_spectral_grids, degraded_bytes, image_feature


def build_feature_rows(model: torch.nn.Module, device: torch.device) -> tuple[list[dict[str, object]], dict[str, object]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grids = build_spectral_grids()
    rows: list[dict[str, object]] = []
    target_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    seen_images: set[str] = set()
    degraded_index = 0
    malformed = 0
    IMAGE_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_id, shard_name, url in SHARDS:
        path = RAW_DIR / shard_name
        download_shard(url, path)
        parquet = pq.ParquetFile(path)
        columns = [
            column
            for column in ("image_name", "image_data", "model_name", "prompt", "real_source", "subset", "split", "label")
            if column in parquet.schema_arrow.names
        ]
        if "image_data" not in columns or "model_name" not in columns:
            raise ValueError(f"{shard_name} does not have image_data/model_name columns")
        print(f"Scanning {shard_name}: {parquet.metadata.num_rows:,} rows")
        for batch in parquet.iter_batches(batch_size=8, columns=columns):
            for record in batch.to_pylist():
                classified = label_record(record)
                if classified is None:
                    continue
                label, is_ood = classified
                model_key = normalize_model(record.get("model_name"))
                if is_ood:
                    if unknown_counts[model_key] >= MAX_UNKNOWN_PER_MODEL or sum(unknown_counts.values()) >= MAX_UNKNOWN_TOTAL:
                        continue
                elif target_counts[label] >= MAX_PER_CLASS:
                    continue
                try:
                    encoded = image_bytes(record.get("image_data"))
                    digest = hashlib.sha256(encoded).hexdigest()
                    if digest in seen_images:
                        continue
                    with Image.open(io.BytesIO(encoded)) as check:
                        check.verify()
                    seen_images.add(digest)
                    prompt = str(record.get("prompt") or "")
                    image_path = None
                    if not is_ood:
                        image_path = IMAGE_CANDIDATE_DIR / label / f"{digest}.image"
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        image_path.write_bytes(encoded)
                    common = {
                        "label": label,
                        "is_ood": is_ood,
                        "sample_kind": "unsupported_generator" if is_ood else "clean",
                        "group_id": group_id(prompt, encoded),
                        "image_name": str(record.get("image_name") or ""),
                        "model_name": str(record.get("model_name") or ""),
                        "prompt": prompt,
                        "real_source": str(record.get("real_source") or ""),
                        "subset": str(record.get("subset") or ""),
                        "source_shard": shard_name,
                        "source_split": str(record.get("split") or ""),
                        "source_label": str(record.get("label") or ""),
                        "dataset_id": dataset_id,
                        "dataset_kind": "communityforensics",
                        "image_path": str(image_path) if image_path else "",
                    }
                    flat, visual, scalars = image_feature(encoded, model, device, grids)
                    rows.append({**common, "flat_spectrum": flat, "visual_features": visual, "scalar_12": scalars})
                    if is_ood:
                        unknown_counts[model_key] += 1
                    else:
                        target_counts[label] += 1
                        degraded, degradation_kind = degraded_bytes(encoded, degraded_index)
                        degraded_index += 1
                        d_flat, d_visual, d_scalars = image_feature(degraded, model, device, grids)
                        rows.append(
                            {
                                **common,
                                "is_ood": True,
                                "sample_kind": degradation_kind,
                                "flat_spectrum": d_flat,
                                "visual_features": d_visual,
                                "scalar_12": d_scalars,
                            }
                        )
                except Exception as exc:
                    malformed += 1
                    if malformed <= 20:
                        print(f"Skipped {shard_name}/{record.get('image_name', '?')}: {exc}")
        path.unlink(missing_ok=True)
        print(f"Finished {shard_name}; classes={dict(target_counts)}; unknown={sum(unknown_counts.values())}")

    missing = [label for label in TARGET_LABELS if target_counts[label] < 2]
    if missing or degraded_index < 2:
        raise ValueError(
            f"Not enough usable rows; missing classes={missing}; "
            f"degraded OOD examples={degraded_index}"
        )
    feature_summary = {
        "dataset_id": "OwensLab/CommunityForensics + OwensLab/CommunityForensics-Small",
        "target_counts": dict(target_counts),
        "unknown_counts": dict(unknown_counts),
        "unknown_total": int(sum(unknown_counts.values())),
        "degraded_examples": degraded_index,
        "malformed_rows": malformed,
        "source_shards": [name for _, name, _ in SHARDS],
        "sampling": {
            "max_per_target_class": MAX_PER_CLASS,
            "max_unsupported_per_generator": MAX_UNKNOWN_PER_MODEL,
            "max_unsupported_total": MAX_UNKNOWN_TOTAL,
        },
        "original_source_split_counts": {
            split: int(sum(row["source_split"] == split and not row["is_ood"] for row in rows))
            for split in sorted({str(row["source_split"]) for row in rows if row.get("source_split")})
        },
    }
    write_feature_file(rows)
    return rows, feature_summary

def write_feature_file(rows: list[dict[str, object]]) -> None:
    fields = [
        "label", "is_ood", "sample_kind", "group_id", "image_name", "model_name", "prompt",
        "real_source", "subset", "source_shard", "source_split", "source_label", "dataset_id",
        "dataset_kind", "visual_features", "flat_spectrum", "scalar_12", "image_data",
    ]
    table_rows = []
    for row in rows:
        output = {key: row.get(key) for key in fields}
        for feature in ("visual_features", "flat_spectrum", "scalar_12"):
            output[feature] = np.asarray(output[feature], dtype=np.float32).tolist()
        output["image_data"] = None
        table_rows.append(output)
    table = pa.Table.from_pylist(table_rows)
    pq.write_table(table, FEATURES_PATH, compression="zstd", row_group_size=16)
    print(f"Feature table: {FEATURES_PATH} ({table.num_rows:,} rows)")

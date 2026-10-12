"""Export a deterministic, held-out image set with source provenance."""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import IMAGE_DIR, IMAGE_EXPORT_PER_CLASS, OUTPUT_DIR, TARGET_LABELS


def write_image_export(
    rows: list[dict[str, object]], test_indices, *, seed: int = 42
) -> tuple[Path, dict[str, int]]:
    """Write exactly N distinct original test images per class and a provenance manifest."""
    if IMAGE_DIR.exists():
        shutil.rmtree(IMAGE_DIR)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, list[dict[str, object]]] = {label: [] for label in TARGET_LABELS}
    for index in test_indices:
        row = rows[int(index)]
        label = str(row.get("label", ""))
        path = str(row.get("image_path", ""))
        if bool(row.get("is_ood")) or label not in candidates or not path:
            continue
        if Path(path).is_file():
            candidates[label].append(row)

    manifest: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for label in TARGET_LABELS:
        ordered = sorted(
            candidates[label],
            key=lambda row: hashlib.sha256(
                f"{seed}:{row.get('group_id', '')}:{row.get('image_name', '')}".encode("utf-8")
            ).hexdigest(),
        )
        selected: list[dict[str, object]] = []
        seen_hashes: set[str] = set()
        for row in ordered:
            image_path = Path(str(row["image_path"]))
            encoded = image_path.read_bytes()
            digest = hashlib.sha256(encoded).hexdigest()
            if digest in seen_hashes:
                continue
            try:
                with Image.open(io.BytesIO(encoded)) as image:
                    image.verify()
            except Exception:
                continue
            seen_hashes.add(digest)
            selected.append({**row, "_encoded": encoded, "_sha256": digest})
            if len(selected) == IMAGE_EXPORT_PER_CLASS:
                break
        if len(selected) != IMAGE_EXPORT_PER_CLASS:
            raise ValueError(
                f"Test split has {len(selected)} usable distinct images for {label}; "
                f"need {IMAGE_EXPORT_PER_CLASS}. Increase the class sample cap or revise the split."
            )

        for position, row in enumerate(selected, start=1):
            encoded = bytes(row["_encoded"])
            try:
                with Image.open(io.BytesIO(encoded)) as image:
                    extension = (image.format or "PNG").lower()
            except Exception as exc:
                raise ValueError(f"Could not read selected test image for {label}: {exc}") from exc
            extension = {"jpeg": "jpg", "tiff": "tif"}.get(extension, extension)
            original_name = Path(str(row.get("image_name") or "image")).stem
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original_name)[:80] or "image"
            filename = f"{position:02d}_{safe_name}.{extension}"
            destination = IMAGE_DIR / label / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(encoded)
            manifest.append(
                {
                    "file": (Path("images") / label / filename).as_posix(),
                    "label": label,
                    "partition": "internal_test",
                    "dataset_id": row.get("dataset_id", ""),
                    "source_shard": row.get("source_shard", ""),
                    "source_split": row.get("source_split", ""),
                    "source_label": row.get("source_label", ""),
                    "model_name": row.get("model_name", ""),
                    "real_source": row.get("real_source", ""),
                    "subset": row.get("subset", ""),
                    "original_image_name": row.get("image_name", ""),
                    "group_id": row.get("group_id", ""),
                    "bytes": len(encoded),
                    "sha256": row["_sha256"],
                }
            )
        counts[label] = len(selected)

    manifest_path = OUTPUT_DIR / "communityforensics-image-manifest.csv"
    fields = list(manifest[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    sheet_path = OUTPUT_DIR / "communityforensics-image-contact-sheet.png"
    make_contact_sheet(manifest, IMAGE_DIR, sheet_path)
    return manifest_path, counts


def make_contact_sheet(manifest: list[dict[str, object]], image_root: Path, output: Path) -> None:
    columns, cell_w, image_h, label_h = 5, 240, 210, 42
    rows = math.ceil(len(manifest) / columns)
    canvas = Image.new("RGB", (columns * cell_w, rows * (image_h + label_h)), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for index, item in enumerate(manifest):
        relative = Path(str(item["file"]).replace("images/", "", 1))
        with Image.open(image_root / relative) as original:
            image = original.convert("RGB")
            image.thumbnail((cell_w - 12, image_h - 8), Image.Resampling.LANCZOS)
        col, row = index % columns, index // columns
        x = col * cell_w + (cell_w - image.width) // 2
        y = row * (image_h + label_h) + (image_h - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text(
            (col * cell_w + 5, row * (image_h + label_h) + image_h + 4),
            f"{item['label']} · {item['model_name']}",
            fill="#222222",
            font=font,
        )
    canvas.save(output, optimize=True)

"""Install previously downloaded model, test-image, and report archives."""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MODEL_ARCHIVE = "genaifp-model-bundle.zip"
IMAGE_ARCHIVE = "genaifp-test-images.zip"
REPORT_ARCHIVE = "genaifp-reports.zip"
MODEL_FILES = {
    "attribution_hybrid.json",
    "attribution_hybrid.pkl",
    "attribution_hybrid.manifest.json",
    "feature_extractor.pkl",
}
REPORT_FILES = {
    "attribution_hybrid.manifest.json",
    "environment.json",
    "evaluation_summary.json",
    "feature_importance.csv",
    "import_summary.json",
}


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        relative = PurePosixPath(member.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe path in downloaded archive: {member.filename}")
        target = (destination / Path(*relative.parts)).resolve()
        if destination != target and destination not in target.parents:
            raise ValueError(f"Archive member escapes install folder: {member.filename}")
    archive.extractall(destination)


def archive_names(archive: zipfile.ZipFile) -> set[str]:
    return {PurePosixPath(name).name for name in archive.namelist() if not name.endswith("/")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=ROOT / "kaggle-output" / "downloads")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--images", type=Path, default=ROOT / "samples" / "communityforensics-test")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--model-archive", type=Path, help="Override model ZIP path")
    parser.add_argument("--image-archive", type=Path, help="Override test-image ZIP path")
    parser.add_argument("--report-archive", type=Path, help="Override reports ZIP path")
    args = parser.parse_args()

    downloads = args.downloads.resolve()
    archives = {
        "model": (args.model_archive or downloads / MODEL_ARCHIVE).resolve(),
        "images": (args.image_archive or downloads / IMAGE_ARCHIVE).resolve(),
        "reports": (args.report_archive or downloads / REPORT_ARCHIVE).resolve(),
    }
    for path in archives.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required archive not found: {path}")

    with zipfile.ZipFile(archives["model"]) as archive:
        if archive_names(archive) != MODEL_FILES or len(archive_names(archive)) != len(MODEL_FILES):
            raise ValueError(f"Model archive contents differ from expected files: {sorted(archive_names(archive))}")
        manifest = json.loads(archive.read("attribution_hybrid.manifest.json"))
        classes = set(manifest.get("classes", []))
        if len(classes) != 12 or "real_camera" not in classes:
            raise ValueError(f"Expected 11 generator labels plus real_camera; found {sorted(classes)}")

    with zipfile.ZipFile(archives["images"]) as archive:
        image_rows = list(csv.DictReader(io.TextIOWrapper(archive.open("manifest.csv"), encoding="utf-8")))
        counts: dict[str, int] = {}
        for row in image_rows:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
            if row.get("partition") != "internal_test":
                raise ValueError(f"Image is not marked as held out: {row.get('file')}")
        image_files = [
            name for name in archive.namelist()
            if name.startswith("images/") and not name.endswith("/")
        ]
        if set(counts) != classes or any(count != 10 for count in counts.values()):
            raise ValueError(f"Expected 10 held-out images per model class; found {counts}")
        if len(image_files) != 120 or len(image_rows) != 120:
            raise ValueError(
                f"Expected 120 image files and manifest rows; found {len(image_files)} and {len(image_rows)}"
            )

    with zipfile.ZipFile(archives["reports"]) as archive:
        names = archive_names(archive)
        if names != REPORT_FILES or len(names) != len(REPORT_FILES):
            raise ValueError(f"Report archive contents differ from expected files: {sorted(names)}")

    args.models.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives["model"]) as archive:
        safe_extract(archive, args.models)
    if args.images.exists():
        shutil.rmtree(args.images)
    args.images.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives["images"]) as archive:
        safe_extract(archive, args.images)
    with zipfile.ZipFile(archives["reports"]) as archive:
        safe_extract(archive, args.reports)

    # Remove the earlier experiment reports that were previously stored here.
    shutil.rmtree(args.reports / "real_data", ignore_errors=True)
    print(f"Installed model files in: {args.models}")
    print(f"Installed 120 labeled held-out images in: {args.images}")
    print(f"Installed current Kaggle reports directly in: {args.reports}")


if __name__ == "__main__":
    main()

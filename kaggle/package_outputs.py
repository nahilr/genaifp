"""Package a completed Kaggle training run into three portable ZIP files."""
from __future__ import annotations

import csv
import io
import shutil
import zipfile
from pathlib import Path

from config import (
    IMAGE_DIR,
    IMAGE_EXPORT_PER_CLASS,
    IMAGE_CANDIDATE_DIR,
    OUTPUT_DIR,
    RAW_DIR,
    TARGET_LABELS,
    WORK,
)

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
MODEL_ARCHIVE = WORK / "genaifp-model-bundle.zip"
IMAGE_ARCHIVE = WORK / "genaifp-test-images.zip"
REPORT_ARCHIVE = WORK / "genaifp-reports.zip"


def _write_zip(destination: Path, files: list[tuple[Path, str]]) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, archive_name in files:
            if not source.is_file():
                raise FileNotFoundError(f"Required run artifact is missing: {source}")
            archive.write(source, archive_name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"Archive validation failed: {destination.name}")
    temporary.replace(destination)


def _validate_existing_archives() -> bool:
    archives = (MODEL_ARCHIVE, IMAGE_ARCHIVE, REPORT_ARCHIVE)
    if not all(path.is_file() for path in archives):
        return False
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"Existing archive is corrupt: {path}")
    return True


def package_outputs() -> None:
    """Package a completed training stage; safe to repeat after a successful package."""
    if not OUTPUT_DIR.is_dir():
        if _validate_existing_archives():
            print("The three Kaggle output archives are already packaged and valid.")
            return
        raise FileNotFoundError(
            f"No training artifacts found at {OUTPUT_DIR}; run the training stage first."
        )

    model_paths = [(OUTPUT_DIR / name, name) for name in sorted(MODEL_FILES)]
    report_paths = [(OUTPUT_DIR / name, name) for name in sorted(REPORT_FILES)]
    _write_zip(MODEL_ARCHIVE, model_paths)
    _write_zip(REPORT_ARCHIVE, report_paths)

    image_manifest = OUTPUT_DIR / "communityforensics-image-manifest.csv"
    contact_sheet = OUTPUT_DIR / "communityforensics-image-contact-sheet.png"
    image_files = [
        (path, (Path("images") / path.relative_to(IMAGE_DIR)).as_posix())
        for path in sorted(IMAGE_DIR.rglob("*"))
        if path.is_file()
    ]
    image_files.extend(
        [
            (image_manifest, "manifest.csv"),
            (contact_sheet, "contact_sheet.png"),
        ]
    )
    _write_zip(IMAGE_ARCHIVE, image_files)
    with zipfile.ZipFile(IMAGE_ARCHIVE) as archive:
        image_names = [
            name for name in archive.namelist()
            if name.startswith("images/") and not name.endswith("/")
        ]
        rows = list(csv.DictReader(io.TextIOWrapper(archive.open("manifest.csv"), encoding="utf-8")))
        counts = {label: 0 for label in TARGET_LABELS}
        for row in rows:
            if row.get("label") in counts and row.get("partition") == "internal_test":
                counts[row["label"]] += 1
        expected = len(TARGET_LABELS) * IMAGE_EXPORT_PER_CLASS
        if len(image_names) != expected or len(rows) != expected or any(
            count != IMAGE_EXPORT_PER_CLASS for count in counts.values()
        ):
            raise RuntimeError(
                f"Test image archive must contain {IMAGE_EXPORT_PER_CLASS} held-out images per class; "
                f"found {len(image_names)} files and counts {counts}"
            )

    shutil.rmtree(RAW_DIR, ignore_errors=True)
    shutil.rmtree(IMAGE_CANDIDATE_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    for path in (MODEL_ARCHIVE, IMAGE_ARCHIVE, REPORT_ARCHIVE):
        print(f"Packaged: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def run_package_outputs() -> None:
    package_outputs()


if __name__ == "__main__":
    run_package_outputs()

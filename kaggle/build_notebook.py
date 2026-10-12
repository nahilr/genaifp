"""Build the uploadable Kaggle notebook from the function-organized source files."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "communityforensics_train.ipynb"
SECTIONS = [
    ("Run settings", "config.py"),
    ("Dataset access and labels", "data_access.py"),
    ("Image and feature extraction", "features.py"),
    ("Original-image sample export", "image_export.py"),
    ("Dataset scan and feature table", "dataset_build.py"),
    ("Training, calibration, and evaluation", "training.py"),
    ("Training stage", "main.py"),
    ("Packaging stage", "package_outputs.py"),
]
SIBLING_IMPORT = re.compile(
    r"^from (?:config|data_access|features|image_export|dataset_build|training) import .*\n",
    re.MULTILINE,
)


def source_cell(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    source = re.sub(
        r"^from (?:config|data_access|features|image_export|dataset_build|training) import \(\n"
        r"(?:[ \t].*\n)*?^\)\n",
        "",
        source,
        flags=re.MULTILINE,
    )
    source = SIBLING_IMPORT.sub("", source)
    source = re.sub(
        r"^if __name__ == [\"']__main__[\"']:\n    [A-Za-z_][A-Za-z0-9_]*\(\)\n?",
        "",
        source,
        flags=re.MULTILINE,
    )
    return source.rstrip() + "\n"


def cell(kind: str, text: str) -> dict[str, object]:
    return {
        "cell_type": kind,
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest()[:8],
        "metadata": {},
        "source": text.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


def main() -> None:
    cells = [cell("markdown", "# CommunityForensics workflow\n\nThis standalone Kaggle notebook has separate stages. First, it downloads the configured CommunityForensics shards, extracts features, creates prompt-grouped train/calibration/test partitions, trains and evaluates the 11 generator labels plus `real_camera`, and selects 10 held-out images per class. Second, it packages the model, test images, and generated reports into separate ZIP files. Heavy dataset processing and training run in Kaggle; local computers only need to download and install those outputs.\n\nSource code is organized by function in adjacent `.py` files and embedded here for Kaggle execution. The notebook does not import the local application package or install this repository. Commercial rows marked `test` are used in an internal grouped evaluation, not an official benchmark score. Run the training stage before packaging; the packaging stage can be rerun while the training artifacts remain in the Kaggle session.\n")]
    for heading, filename in SECTIONS:
        cells.append(cell("markdown", f"## {heading}\n"))
        cells.append(cell("code", source_cell(ROOT / filename)))
    cells.append(cell("markdown", "## Run the training stage\n\nRun this cell to download/process the data, train/evaluate the model, and leave lightweight artifacts ready for packaging.\n"))
    cells.append(cell("code", "main()\n"))
    cells.append(cell("markdown", "## Package the outputs\n\nRun this after training. It creates separate model, test-image, and report archives. It can also be run on its own after a completed training stage in this Kaggle session.\n"))
    cells.append(cell("code", "package_outputs()\n"))
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()

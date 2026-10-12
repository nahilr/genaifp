"""Download the three packaged outputs from a completed Kaggle kernel run."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "nahilr/genaifp"
ARCHIVES = (
    "genaifp-model-bundle.zip",
    "genaifp-test-images.zip",
    "genaifp-reports.zip",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default=KERNEL, help="Kaggle kernel handle")
    parser.add_argument("--output", type=Path, default=ROOT / "kaggle-output")
    args = parser.parse_args()

    download_dir = args.output.resolve() / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    kaggle = shutil.which("kaggle")
    command = ([kaggle] if kaggle else [sys.executable, "-m", "kaggle"])
    command += [
        "kernels", "output", args.kernel,
        "--path", str(download_dir),
        "--file-pattern", r"^(genaifp-model-bundle|genaifp-test-images|genaifp-reports)\.zip$",
        "--force",
    ]

    for attempt in range(3):
        for name in ARCHIVES:
            (download_dir / name).unlink(missing_ok=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            missing = [name for name in ARCHIVES if not (download_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Kaggle did not return expected archives: {missing}")
            for name in ARCHIVES:
                with zipfile.ZipFile(download_dir / name) as archive:
                    if archive.testzip() is not None:
                        raise ValueError(f"Downloaded archive is corrupt: {name}")
            break
        except (subprocess.CalledProcessError, OSError, zipfile.BadZipFile, ValueError) as exc:
            if attempt == 2:
                raise
            print(f"Kaggle download attempt {attempt + 1} failed ({exc}); retrying shortly...", flush=True)
            time.sleep(3 * (attempt + 1))

    print(f"Downloaded model, test image, and report archives to: {download_dir}")


if __name__ == "__main__":
    main()

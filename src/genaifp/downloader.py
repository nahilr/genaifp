"""Downloader utilities for CommunityForensics dataset shards and curated subsets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

SHARDS = {
    "HFCF_Commercial_0.parquet": {
        "url": "https://huggingface.co/datasets/OwensLab/CommunityForensics/resolve/main/data/commercial/HFCF_Commercial_0.parquet",
        "size_gb": 5.64,
        "models": [
            "Dalle3",
            "FLUX-dev",
            "FLUX-schnell",
            "IdeogramV1",
            "Firefly_Image2",
            "Firefly_Image3",
        ],
    },
    "HFCF_Commercial_1.parquet": {
        "url": "https://huggingface.co/datasets/OwensLab/CommunityForensics/resolve/main/data/commercial/HFCF_Commercial_1.parquet",
        "size_gb": 10.94,
        "models": ["MidjourneyV6_1", "MidjourneyV5_2", "Imagen3", "IdeogramV2", "Dalle2"],
    },
}

REAL_SHARDS = {
    f"HFCF_small_{i}.parquet": {
        "url": f"https://huggingface.co/datasets/OwensLab/CommunityForensics-Small/resolve/main/data/HFCF_small_{i}.parquet",
        "size_gb": 1.78,
    }
    for i in range(174, 186)
}

TARGET_ENGINES = {
    "dalle2": {"source": "commercial_1", "filter_model": "Dalle2"},
    "dalle3": {"source": "commercial_0", "filter_model": "Dalle3"},
    "midjourney_v5_2": {"source": "commercial_1", "filter_model": "MidjourneyV5_2"},
    "midjourney_v6_1": {"source": "commercial_1", "filter_model": "MidjourneyV6_1"},
    "firefly_image2": {"source": "commercial_0", "filter_model": "Firefly_Image2"},
    "firefly_image3": {"source": "commercial_0", "filter_model": "Firefly_Image3"},
    "flux_dev": {"source": "commercial_0", "filter_model": "FLUX-dev"},
    "flux_schnell": {"source": "commercial_0", "filter_model": "FLUX-schnell"},
    "ideogram_v1": {"source": "commercial_0", "filter_model": "IdeogramV1"},
    "ideogram_v2": {"source": "commercial_1", "filter_model": "IdeogramV2"},
    "imagen3": {"source": "commercial_1", "filter_model": "Imagen3"},
    "real_camera": {"source": "real", "filter_model": "LandscapesHQ"},
}

RANGE_CHUNK_SIZE = 64 * 1024 * 1024
RANGE_ATTEMPTS = 5


def download_range(
    url: str,
    start: int,
    end: int,
    session: requests.Session | None,
    output_path: Path,
    progress_bar: tqdm = None,
) -> int:
    """Download a byte range with retries; failed attempts are overwritten on retry."""
    headers = {"Range": f"bytes={start}-{end}"}
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(RANGE_ATTEMPTS):
        local_session = session or requests.Session()
        response = None
        try:
            response = local_session.get(url, headers=headers, stream=True, timeout=(30, 90))
            response.raise_for_status()
            if response.status_code != 206:
                raise OSError(
                    f"Server ignored byte range {start}-{end} (HTTP {response.status_code})"
                )
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-{end}/"):
                raise OSError(f"Unexpected Content-Range for bytes {start}-{end}: {content_range!r}")
            bytes_written = 0
            with open(output_path, "r+b") as f:
                f.seek(start)
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
            if bytes_written != expected:
                raise OSError(
                    f"Downloaded {bytes_written} bytes for range {start}-{end}; expected {expected}"
                )
            if progress_bar:
                progress_bar.update(bytes_written)
            return bytes_written
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt + 1 < RANGE_ATTEMPTS:
                time.sleep(min(2**attempt, 8))
        finally:
            if response is not None:
                response.close()
            if session is None:
                local_session.close()
    raise OSError(f"Range {start}-{end} failed after {RANGE_ATTEMPTS} attempts: {last_error}")


def download_shard(
    url: str, output_path: Path, n_threads: int = 16, skip_existing: bool = False
) -> bool:
    """Download a single shard using multi-threaded range requests."""
    if n_threads < 1 or n_threads > 64:
        raise ValueError("threads must be between 1 and 64")
    with requests.Session() as session:
        head_resp = session.head(url, allow_redirects=True, timeout=30)
        head_resp.raise_for_status()
        if not head_resp.headers.get("Content-Length"):
            raise OSError(f"Server did not provide Content-Length for {url}")
        file_size = int(head_resp.headers["Content-Length"])
        final_url = head_resp.url
    if skip_existing and output_path.is_file() and output_path.stat().st_size == file_size:
        print(f"Skipping {output_path.name} (size matches remote file)")
        return True

    print(f"Downloading {output_path.name} ({file_size / 1e9:.2f} GB) with {n_threads} threads...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".part")
    progress_path = output_path.with_name(output_path.name + ".part.progress.json")
    ranges = [
        (start, min(start + RANGE_CHUNK_SIZE, file_size) - 1)
        for start in range(0, file_size, RANGE_CHUNK_SIZE)
    ]
    metadata = {
        "source_url": url,
        "file_size": file_size,
        "chunk_size": RANGE_CHUNK_SIZE,
        "completed": [],
    }
    if partial_path.is_file() and progress_path.is_file():
        try:
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if all(saved.get(key) == metadata[key] for key in ("source_url", "file_size", "chunk_size")):
                metadata["completed"] = saved.get("completed", [])
        except (OSError, json.JSONDecodeError):
            pass
    if not partial_path.is_file() or not progress_path.is_file() or not metadata["completed"]:
        # A previous partial without a matching progress file cannot be trusted.
        with open(partial_path, "wb") as f:
            f.truncate(file_size)
        metadata["completed"] = []
    elif partial_path.stat().st_size != file_size:
        with open(partial_path, "wb") as f:
            f.truncate(file_size)
        metadata["completed"] = []

    completed = {int(index) for index in metadata["completed"]}

    def save_progress() -> None:
        temporary_progress = progress_path.with_suffix(progress_path.suffix + ".tmp")
        temporary_progress.write_text(json.dumps({**metadata, "completed": sorted(completed)}))
        temporary_progress.replace(progress_path)

    save_progress()

    start_time = time.time()
    with tqdm(
            total=file_size,
            initial=sum(ranges[index][1] - ranges[index][0] + 1 for index in completed),
            unit="B",
            unit_scale=True,
            desc=output_path.name,
            leave=True,
        ) as pbar:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {
                executor.submit(
                    download_range, final_url, start, end, None, partial_path, pbar
                ): index
                for index, (start, end) in enumerate(ranges)
                if index not in completed
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Thread failed: {e}")
                    return False
                completed.add(index)
                save_progress()

    if partial_path.stat().st_size != file_size:
        partial_path.unlink(missing_ok=True)
        print(f"Download size mismatch for {output_path.name}")
        return False
    partial_path.replace(output_path)
    progress_path.unlink(missing_ok=True)

    elapsed = time.time() - start_time
    speed_mb_s = (file_size / 1e6) / max(elapsed, 0.001)
    print(f"  Done in {elapsed:.1f}s ({speed_mb_s:.1f} MB/s)")
    return True


def download_shards_main(args=None):
    parser = argparse.ArgumentParser(description="Download CommunityForensics shards")
    parser.add_argument("--output_dir", default="data/raw", help="Output directory")
    parser.add_argument(
        "--shards",
        nargs="+",
        choices=["commercial", "real", "all"],
        default=["commercial"],
        help="Which shards to download",
    )
    parser.add_argument("--threads", type=int, default=16, help="Download threads")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing")
    parsed = parser.parse_args(args)

    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards_to_download = {}
    if "commercial" in parsed.shards or "all" in parsed.shards:
        shards_to_download.update(SHARDS)
    if "real" in parsed.shards or "all" in parsed.shards:
        shards_to_download.update(REAL_SHARDS)
    if not shards_to_download:
        print("ERROR: no shard groups selected", file=sys.stderr)
        return 1

    for name, info in shards_to_download.items():
        output_path = output_dir / name
        try:
            if not download_shard(
                info["url"],
                output_path,
                n_threads=parsed.threads,
                skip_existing=parsed.skip_existing,
            ):
                return 1
        except Exception as exc:
            print(f"Failed to download {name}: {exc}", file=sys.stderr)
            return 1
    return 0


def download_curated_main(args=None):
    parser = argparse.ArgumentParser(description="Curated image downloads are not supported yet")
    parser.add_argument("--output_dir", default="data/curated")
    parser.add_argument("--max_per_class", type=int, default=250)
    parser.parse_args(args)
    print(
        "ERROR: curated image downloads are not implemented. Download CommunityForensics Parquet "
        "shards with `download_shards`, then build features with `build`.",
        file=sys.stderr,
    )
    return 1

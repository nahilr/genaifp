"""Dataset labels, image-payload helpers, and resumable Parquet shard downloads."""
from __future__ import annotations

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests

from config import RANGE_CHUNK_BYTES


def normalize_model(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").split("/")[-1].lower())

def image_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict) and value.get("bytes"):
        return image_bytes(value["bytes"])
    raise ValueError(f"Unsupported or empty image_data payload: {type(value).__name__}")

def label_record(record: dict[str, object]) -> tuple[str, bool] | None:
    model = normalize_model(record.get("model_name"))
    source_label = str(record.get("label", ""))
    if model in TARGET_MODELS:
        return TARGET_MODELS[model], False
    if model in REAL_MODELS or source_label == "0":
        return "real_camera", False
    if source_label == "1" and model:
        return "unknown", True
    return None

def download_range(url: str, start: int, end: int, path: Path) -> None:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise OSError(f"Unexpected range response: {response.status_code} {content_range}")
                written = 0
                with path.open("r+b") as handle:
                    handle.seek(start)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            written += len(chunk)
                if written != expected:
                    raise OSError(f"Range wrote {written} bytes; expected {expected}")
                return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 8))
    raise OSError(f"Range {start}-{end} failed after retries: {last_error}")

def download_shard(url: str, output_path: Path) -> None:
    with requests.Session() as session:
        response = session.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        size = int(response.headers["Content-Length"])
        final_url = response.url
    if output_path.is_file() and output_path.stat().st_size == size:
        try:
            pq.ParquetFile(output_path).metadata.num_rows
            print(f"Reusing {output_path.name} ({size / 1e9:.2f} GB)")
            return
        except Exception:
            output_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.truncate(size)
    ranges = [
        (start, min(start + RANGE_CHUNK_BYTES, size) - 1)
        for start in range(0, size, RANGE_CHUNK_BYTES)
    ]
    print(f"Downloading {output_path.name} ({size / 1e9:.2f} GB) in {len(ranges)} ranges")
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(download_range, final_url, start, end, output_path) for start, end in ranges]
        for future in as_completed(futures):
            future.result()
    pq.ParquetFile(output_path).metadata.num_rows

def group_id(prompt: object, encoded: bytes) -> str:
    normalized = " ".join(str(prompt or "").casefold().split())
    value = f"prompt:{normalized}" if normalized else "image:" + hashlib.sha256(encoded).hexdigest()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

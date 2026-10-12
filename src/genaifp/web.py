"""Local browser interface for experimental image attribution."""

from __future__ import annotations

import argparse
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartParser

from .data_pipeline import GROUPING_METHOD
from .feature_extractor import FeatureExtractor
from .preprocessor import preprocess_array
from .spectral_engine import SpectralEngine

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).with_name("web")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
# Keep uploads below the enforced request limit in memory instead of spilling to disk.
MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES


def _artifact_status(model_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    manifest_path = model_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        if model_path.with_suffix(".json").is_file() or model_path.with_suffix(".pkl").is_file():
            return None, (
                "Model files are present, but they are missing the real-data evaluation and "
                "calibration manifest required for analysis. Train and evaluate a model from "
                "CommunityForensics data first."
            )
        return None, "No evaluated real-data model is installed. Train one from CommunityForensics data first."
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "The model provenance manifest is missing or unreadable."
    ready = (
        manifest.get("format_version") == 1
        and manifest.get("dataset_kind") == "communityforensics"
        and manifest.get("evaluation_status") == "research_holdout_complete"
        and manifest.get("evaluated_on_heldout_groups") is True
        and manifest.get("grouping_method") == GROUPING_METHOD
        and manifest.get("probabilities_calibrated") is True
        and manifest.get("model_type") == "hybrid_524d"
        and bool(manifest.get("classes"))
    )
    if not ready:
        return None, "This model is not marked as calibrated and evaluated on held-out real data."
    if not model_path.with_suffix(".json").is_file() or not model_path.with_suffix(".pkl").is_file():
        return None, "The model files listed by its provenance manifest are incomplete."
    feature_path = model_path.parent / str(manifest.get("feature_extractor", ""))
    if not feature_path.is_file():
        return None, "The model's fitted feature transform is missing."
    return {**manifest, "feature_path": str(feature_path)}, None


def _load_runtime(model_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    manifest, error = _artifact_status(model_path)
    if error:
        return None, error
    try:
        from .classifier import AttributionClassifier
        from .visual_engine import VisualEngine

        classifier = AttributionClassifier.load(model_path)
        extractor = FeatureExtractor.load(manifest["feature_path"])
        if (
            not classifier.probabilities_calibrated
            or classifier.ood_detector is None
            or getattr(classifier.model, "n_features_in_", None) != 524
            or list(classifier.class_names) != list(manifest.get("classes", []))
        ):
            return None, "The model files do not match their calibrated provenance manifest."
        visual = VisualEngine(crop_size=int(manifest.get("preprocessing", {}).get("crop_size", 512)))
        return {
            "classifier": classifier,
            "extractor": extractor,
            "visual": visual,
            "spectral": SpectralEngine(
                crop_size=int(manifest.get("preprocessing", {}).get("crop_size", 512))
            ),
            "manifest": manifest,
        }, None
    except Exception:
        return None, "The evaluated model could not be loaded. Check its files and vision dependencies."


def analyze_image(image: Image.Image, runtime: dict[str, Any]) -> dict[str, Any]:
    """Run the same center crop through the visual and spectral branches."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    crop_size = runtime["spectral"].crop_size
    residual = preprocess_array(rgb, crop_size=crop_size, allow_pad=True)
    spectral = runtime["spectral"].process(residual)
    features = runtime["extractor"].extract(spectral)
    visual = runtime["visual"].extract(rgb)
    vector = np.concatenate(
        [visual, features["pca_features"], features["scalar_12"]]
    ).astype(np.float32)
    result = runtime["classifier"].predict_single(vector)
    result["model_provenance"] = {
        "dataset_id": runtime["manifest"].get("dataset_id"),
        "evaluation_status": runtime["manifest"].get("evaluation_status"),
        "held_out_test_samples": runtime["manifest"].get("sample_counts", {}).get("test_known"),
        "dataset_sha256": runtime["manifest"].get("dataset_sha256"),
        "grouping_method": runtime["manifest"].get("grouping_method"),
        "grouping_limitations": runtime["manifest"].get("grouping_limitations", []),
    }
    result["notice"] = "Experimental research output. This result is not forensic evidence."
    return result


def create_app(model_path: str | Path | None = None) -> FastAPI:
    base = Path(
        model_path
        or os.environ.get("GENAIFP_MODEL", ROOT / "models/attribution_hybrid")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime, app.state.not_ready_reason = _load_runtime(base)
        yield
        app.state.runtime = None

    app = FastAPI(
        title="GenAI Fingerprint",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")

    @app.middleware("http")
    async def local_request_safety(request, call_next):
        allow_remote = os.environ.get("GENAIFP_ALLOW_REMOTE", "").lower() in ("1", "true", "yes")
        if not allow_remote and request.url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return JSONResponse({"detail": "This service accepts local browser requests only."}, status_code=403)
        if request.url.path == "/api/predict":
            length = request.headers.get("content-length")
            if length:
                try:
                    request_length = int(length)
                except ValueError:
                    return JSONResponse({"detail": "Invalid upload request length."}, status_code=400)
                if request_length > MAX_UPLOAD_BYTES + 64 * 1024:
                    return JSONResponse({"detail": "Upload request exceeds the 20 MB limit."}, status_code=413)
            origin = request.headers.get("origin")
            if origin and not allow_remote:
                parsed_origin = urlsplit(origin)
                if (
                    parsed_origin.scheme != request.url.scheme
                    or parsed_origin.netloc.lower() != request.headers.get("host", "").lower()
                ):
                    return JSONResponse({"detail": "Cross-origin uploads are not accepted."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(ASSETS / "index.html")

    @app.get("/about", include_in_schema=False)
    async def about() -> FileResponse:
        return FileResponse(ASSETS / "about.html")

    @app.get("/api/health")
    async def health() -> JSONResponse:
        runtime = getattr(app.state, "runtime", None)
        response = {"ready": runtime is not None}
        if runtime is None:
            response["message"] = getattr(
                app.state,
                "not_ready_reason",
                "Model readiness has not been checked yet.",
            )
        else:
            response["provenance"] = {
                "dataset_id": runtime["manifest"].get("dataset_id"),
                "evaluation_status": runtime["manifest"].get("evaluation_status"),
            }
        return JSONResponse(response)

    @app.post("/api/predict")
    async def predict(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        runtime = getattr(app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail=getattr(app.state, "not_ready_reason", "No evaluated model is ready."),
            )
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if not payload:
            raise HTTPException(status_code=400, detail="Choose an image to analyze.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 20 MB upload limit.")
        try:
            with Image.open(io.BytesIO(payload)) as opened:
                if opened.format not in ALLOWED_FORMATS:
                    raise HTTPException(status_code=415, detail="Use a PNG, JPEG, or WebP image.")
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="Image dimensions exceed the 40-megapixel limit.")
                opened.load()
                image = opened.convert("RGB")
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid supported image.") from exc
        return await run_in_threadpool(analyze_image, image, runtime)

    return app


app = create_app()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local GenAI Fingerprint web app")
    parser.add_argument("--model", default=None, help="Model path without .json/.pkl extension")
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Host address to bind to (defaults to 127.0.0.1 or $HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to listen on (defaults to 8000 or $PORT)",
    )
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(create_app(args.model), host=args.host, port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

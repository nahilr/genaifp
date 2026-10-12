"""Local web app readiness, upload validation, and result tests."""

import io

import pytest
from PIL import Image

TestClient = pytest.importorskip("fastapi.testclient").TestClient

web = pytest.importorskip("genaifp.web")


def _image_bytes(format: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 24), (41, 72, 63)).save(output, format=format)
    return output.getvalue()


def test_no_real_manifest_displays_setup_state_and_refuses_prediction(tmp_path):
    client = TestClient(web.create_app(tmp_path / "missing-model"), base_url="http://127.0.0.1")
    with client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ready"] is False
        assert "evaluated real-data model" in health.json()["message"]
        page = client.get("/")
        assert "Model unavailable" in page.text
        response = client.post(
            "/api/predict", files={"file": ("tiny.png", _image_bytes(), "image/png")}
        )
        assert response.status_code == 503


def test_present_legacy_model_files_explain_why_they_are_not_ready(tmp_path):
    model_path = tmp_path / "legacy-model"
    model_path.with_suffix(".json").write_text("{}", encoding="utf-8")
    client = TestClient(web.create_app(model_path), base_url="http://127.0.0.1")
    with client:
        health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ready"] is False
    assert "Model files are present" in health.json()["message"]
    assert "evaluation and calibration manifest" in health.json()["message"]


def test_upload_validation_and_prediction_response(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "_load_runtime", lambda path: ({"fake": True}, None))
    monkeypatch.setattr(
        web,
        "analyze_image",
        lambda image, runtime: {"predicted_engine": "fixture", "class_probabilities": {"fixture": 1.0}},
    )
    client = TestClient(web.create_app(tmp_path / "valid-model"), base_url="http://127.0.0.1")
    with client:
        invalid = client.post(
            "/api/predict", files={"file": ("bad.png", b"not an image", "image/png")}
        )
        assert invalid.status_code == 400
        valid = client.post(
            "/api/predict", files={"file": ("tiny.png", _image_bytes(), "image/png")}
        )
        assert valid.status_code == 200
        assert valid.json()["predicted_engine"] == "fixture"


@pytest.mark.parametrize(
    ("format", "content_type"),
    [("JPEG", "image/jpeg"), ("WEBP", "image/webp")],
)
def test_accepts_supported_formats(monkeypatch, tmp_path, format, content_type):
    monkeypatch.setattr(web, "_load_runtime", lambda path: ({"fake": True}, None))
    monkeypatch.setattr(
        web,
        "analyze_image",
        lambda image, runtime: {"predicted_engine": "fixture"},
    )
    client = TestClient(web.create_app(tmp_path / "valid-model"), base_url="http://127.0.0.1")
    with client:
        response = client.post(
            "/api/predict",
            files={"file": (f"tiny.{format.lower()}", _image_bytes(format), content_type)},
        )
    assert response.status_code == 200
    assert response.json()["predicted_engine"] == "fixture"


def test_rejects_oversized_request_before_multipart_parsing(tmp_path):
    client = TestClient(web.create_app(tmp_path / "missing-model"), base_url="http://127.0.0.1")
    with client:
        response = client.post(
            "/api/predict",
            headers={"Content-Length": str(web.MAX_UPLOAD_BYTES + 64 * 1024 + 1)},
            content=b"",
        )
    assert response.status_code == 413


def test_rejects_cross_origin_uploads(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "_load_runtime", lambda path: ({"fake": True}, None))
    client = TestClient(web.create_app(tmp_path / "valid-model"), base_url="http://127.0.0.1")
    with client:
        response = client.post(
            "/api/predict",
            headers={"Origin": "http://attacker.invalid", "Host": "127.0.0.1"},
            files={"file": ("tiny.png", _image_bytes(), "image/png")},
        )
        assert response.status_code == 403

"""Downloader range integrity and failure propagation."""

import pytest

from genaifp import downloader


class _Response:
    def __init__(self, payload=b"abc", status=206, content_range="bytes 0-2/3"):
        self.payload = payload
        self.status_code = status
        self.headers = {"Content-Range": content_range}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")

    def iter_content(self, chunk_size):
        yield self.payload

    def close(self):
        pass


class _Session:
    def __init__(self, response):
        self.response = response

    def get(self, *args, **kwargs):
        return self.response


def test_range_download_checks_content_range_and_byte_count(tmp_path):
    destination = tmp_path / "partial"
    destination.write_bytes(b"\0\0\0")
    written = downloader.download_range(
        "https://example.invalid/file", 0, 2, _Session(_Response()), destination
    )
    assert written == 3
    assert destination.read_bytes() == b"abc"


def test_range_download_rejects_server_that_ignores_range(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader.time, "sleep", lambda *_: None)
    destination = tmp_path / "partial"
    destination.write_bytes(b"\0\0\0")
    with pytest.raises(IOError, match="ignored byte range"):
        downloader.download_range(
            "https://example.invalid/file", 0, 2, _Session(_Response(status=200)), destination
        )


def test_range_download_retries_transient_connection_failure(monkeypatch, tmp_path):
    class FlakySession:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise downloader.requests.ConnectionError("temporary disconnect")
            return _Response()

    monkeypatch.setattr(downloader.time, "sleep", lambda *_: None)
    destination = tmp_path / "partial"
    destination.write_bytes(b"\0\0\0")
    session = FlakySession()
    written = downloader.download_range(
        "https://example.invalid/file", 0, 2, session, destination
    )
    assert written == 3
    assert session.calls == 2
    assert destination.read_bytes() == b"abc"


def test_shard_command_propagates_download_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        downloader,
        "SHARDS",
        {"bad.parquet": {"url": "https://example.invalid/bad"}},
    )
    monkeypatch.setattr(downloader, "download_shard", lambda *args, **kwargs: False)
    result = downloader.download_shards_main(
        ["--output_dir", str(tmp_path), "--shards", "commercial"]
    )
    assert result == 1


def test_curated_stub_fails_instead_of_claiming_success(capsys):
    assert downloader.download_curated_main([]) == 1
    assert "not implemented" in capsys.readouterr().err

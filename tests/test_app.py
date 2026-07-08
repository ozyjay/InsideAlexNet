from pathlib import Path

import app as demo_app
from app import RunRequest


def test_index_html_mentions_fastapi_demo_title() -> None:
    html = demo_app.index()

    assert "How Does a Neural Network See?" in html
    assert "trained vision model" in html


def test_list_images_handles_empty_folder(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(demo_app, "DEMO_IMAGES_DIR", tmp_path)

    response = demo_app.list_images()

    assert response["images"] == []
    assert "No curated images" in response["empty_message"]


def test_fallback_run_returns_graceful_placeholder(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"discovery only")
    monkeypatch.setattr(demo_app, "DEMO_IMAGES_DIR", tmp_path)

    response = demo_app.run_demo(RunRequest(image_name="sample.jpg", fallback=True))

    assert response.status_code == 200
    assert b"Fallback replay assets" in response.body

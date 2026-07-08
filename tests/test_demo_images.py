from pathlib import Path

from src.demo_images import discover_images, format_image_label, has_demo_images


def test_discover_images_returns_supported_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "zebra.PNG").write_bytes(b"not a real image for discovery only")
    (tmp_path / "apple.jpg").write_bytes(b"not a real image for discovery only")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "inside.jpg").write_bytes(b"ignored because discovery is non-recursive")

    discovered = discover_images(tmp_path)

    assert [path.name for path in discovered] == ["apple.jpg", "zebra.PNG"]


def test_discover_images_handles_missing_and_empty_directories(tmp_path: Path) -> None:
    assert discover_images(tmp_path / "missing") == []
    assert discover_images(tmp_path) == []
    assert has_demo_images(tmp_path) is False


def test_format_image_label_is_readable() -> None:
    assert format_image_label(Path("assets/demo_images/red-fox_example.jpg")) == "Red Fox Example"

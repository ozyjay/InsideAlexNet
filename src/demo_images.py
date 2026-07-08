"""Curated image discovery helpers for the Open Day demo."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEMO_IMAGES_DIR = Path("assets/demo_images")


def discover_images(directory: Path | str = DEMO_IMAGES_DIR) -> list[Path]:
    """Return supported curated images in a stable display order.

    Missing or empty directories are treated as an empty image list so the app
    can explain the next step instead of crashing.
    """
    image_dir = Path(directory)
    if not image_dir.exists() or not image_dir.is_dir():
        return []

    return sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def format_image_label(path: Path) -> str:
    """Format a curated image filename for a simple selector label."""
    return path.stem.replace("_", " ").replace("-", " ").strip().title() or path.name


def has_demo_images(directory: Path | str = DEMO_IMAGES_DIR) -> bool:
    """Return whether the curated image directory contains supported images."""
    return bool(discover_images(directory))

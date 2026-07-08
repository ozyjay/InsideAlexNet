"""Fallback asset helpers for replay mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FALLBACK_DIR = Path("assets/fallback")


def load_json(path: Path | str) -> dict[str, Any]:
    """Load a fallback JSON file, returning an empty mapping when missing.

    Phase 5 will define the final fallback schema. This helper is intentionally
    conservative so the app can keep running when replay assets are absent.
    """
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected fallback JSON object in {json_path}")
    return data

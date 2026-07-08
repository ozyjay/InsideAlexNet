import json
from pathlib import Path

from src.fallback import load_json


def test_load_json_returns_empty_mapping_for_missing_file(tmp_path: Path) -> None:
    assert load_json(tmp_path / "missing.json") == {}


def test_load_json_reads_mapping(tmp_path: Path) -> None:
    path = tmp_path / "fallback.json"
    path.write_text(json.dumps({"image": {"predictions": []}}), encoding="utf-8")

    assert load_json(path) == {"image": {"predictions": []}}

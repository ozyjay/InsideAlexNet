from src.captions import CAPTIONS, get_caption


def test_required_caption_keys_exist() -> None:
    required = {"Input", "Early layer", "Middle layer", "Deep layer", "Prediction"}
    assert required.issubset(CAPTIONS)


def test_unknown_caption_uses_safe_wording() -> None:
    assert "trained vision model" in get_caption("Unknown stage")
